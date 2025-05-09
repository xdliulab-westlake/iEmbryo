import tensorflow as tf 
import re
import time
import jieba
import numpy as np
import pandas as pd
import os
import random
import sentencepiece as spm
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def get_angles(pos, i, d_model):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2*(i // 2))/ np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis,:],
                           d_model)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mark(seq):
    # 获取为0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :] # (batch_size,1,1,seq_len)

def create_look_ahead_mark(size):
    # 1 - 对角线和取下三角的全部对角线（-1->全部）
    # 这样就可以构造出每个时刻未预测token的掩码
    mark = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mark  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention 乘上value
    output = tf.matmul(attention_weights, v) # （.., seq_len_v, depth）

    return output, attention_weights

# 构造mutil head attention层
class MutilHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MutilHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_v, num_heads, depth)

        # 合并多头
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))

        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights
    
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, ddf, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MutilHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, ddf)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask):
        # 多头注意力网络
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, d_model)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2
def point_wise_feed_forward_network(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

class Encoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf,
                input_vocab_size, max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.encode_layer = [EncoderLayer(d_model, n_heads, ddf, drop_rate)
                            for _ in range(n_layers)]

        self.dropout = layers.Dropout(drop_rate)
    def call(self, inputs, training, mark):

        seq_len = inputs.shape[1]
        word_emb = self.embedding(inputs)
        word_emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb = word_emb + self.pos_embedding[:,:seq_len,:]
        x = self.dropout(emb, training=training)
        for i in range(self.n_layers):
            x = self.encode_layer[i](x, training, mark)

        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MutilHeadAttention(d_model, num_heads)
        self.mha2 = MutilHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)
        self.dropout3 = layers.Dropout(drop_rate)

    def call(self,inputs, encode_out, training, 
             look_ahead_mask, padding_mask):
        # masked muti-head attention
        att1, att_weight1 = self.mha1(inputs, inputs, inputs,look_ahead_mask)
        att1 = self.dropout1(att1, training=training)
        out1 = self.layernorm1(inputs + att1)
        # muti-head attention
        att2, att_weight2 = self.mha2(encode_out, encode_out, inputs, padding_mask)
        att2 = self.dropout2(att2, training=training)
        out2 = self.layernorm2(out1 + att2)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.layernorm3(out2 + ffn_out)

        return out3, att_weight1, att_weight2
    
class Decoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf,
                target_vocab_size, max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.decoder_layers= [DecoderLayer(d_model, n_heads, ddf, drop_rate)
                             for _ in range(n_layers)]

        self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, encoder_out,training,
             look_ahead_mark, padding_mark):

        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        h = self.embedding(inputs)
        h *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        h += self.pos_embedding[:,:seq_len,:]

        h = self.dropout(h, training=training)
#         print('--------------------\n',h, h.shape)
        # 叠加解码层
        for i in range(self.n_layers):
            h, att_w1, att_w2 = self.decoder_layers[i](h, encoder_out,
                                                   training, look_ahead_mark,
                                                   padding_mark)
            attention_weights['decoder_layer{}_att_w1'.format(i+1)] = att_w1
            attention_weights['decoder_layer{}_att_w2'.format(i+1)] = att_w2

        return h, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, diff,
                input_vocab_size, target_vocab_size,
                max_seq_len, drop_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, d_model, n_heads,diff,
                              input_vocab_size, max_seq_len, drop_rate)

        self.decoder = Decoder(n_layers, d_model, n_heads, diff,
                              target_vocab_size, max_seq_len, drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    def call(self, inputs, targets, training, encode_padding_mask, 
            look_ahead_mask, decode_padding_mask):

        encode_out = self.encoder(inputs, training, encode_padding_mask)
        print("encode_out", encode_out.shape)
        decode_out, att_weights = self.decoder(targets, encode_out, training, 
                                               look_ahead_mask, decode_padding_mask)
        print("decode_out", decode_out.shape)
        final_out = self.final_layer(decode_out)

        return final_out, att_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32) 
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def create_mask(inputs,targets):
    encode_padding_mask = create_padding_mark(inputs)
    # 这个掩码用于掩输入解码层第二层的编码层输出
    decode_padding_mask = create_padding_mark(inputs)

    # look_ahead 掩码， 掩掉未预测的词
    look_ahead_mask = create_look_ahead_mark(tf.shape(targets)[1])
    # 解码层第一层得到padding掩码
    decode_targets_padding_mask = create_padding_mark(targets)

    # 合并解码层第一层掩码
    combine_mask = tf.maximum(decode_targets_padding_mask, look_ahead_mask)

    return encode_padding_mask, combine_mask, decode_padding_mask

def loss_fun(y_ture, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_ture, 0))  # 为0掩码标1
    loss_ = loss_object(y_ture, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def train_step(inputs, targets):
    tar_inp = targets[:,:-1]
    tar_real = targets[:,1:]
    # 构造掩码
    encode_padding_mask, combined_mask, decode_padding_mask = create_mask(inputs, tar_inp)


    with tf.GradientTape() as tape:
        predictions, _ = transformer(inputs, tar_inp,
                                    True,
                                    encode_padding_mask,
                                    combined_mask,
                                    decode_padding_mask)
        loss = loss_fun(tar_real, predictions)
    # 求梯度
    gradients = tape.gradient(loss, transformer.trainable_variables)
    # 反向传播
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # 记录loss和准确率
    train_loss(loss)
    train_accuracy(tar_real, predictions)

if __name__ == "__main__":
    MAX_LEN = 12
    def load_data(file_path):
        words_re = re.compile(r'\w+')
        corpus = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for k in f:
                en_sent, ch_sent, _ = k.strip("").split("\t")
                if en_sent.find(",") >= 0 or ch_sent.find(",") >= 0:
                    continue
                en_sent = words_re.findall(en_sent.lower())
                ch_sent = ch_sent[:-1]
                if len(en_sent) < MAX_LEN and len(ch_sent) < MAX_LEN:
                    en_sent = " ".join(en_sent)
                    corpus.append([en_sent, ch_sent])
        return corpus

    os.chdir("/Users/lzz/Documents/GitHub/IVF_AI/cmn_eng")
    corpus = load_data('./cmn.txt')

    ### 分词
    corpus_format = []
    for k in corpus:
        en = k[0]
        ch = k[1]
        ch = " ".join(jieba.cut(ch, cut_all=False))
        corpus_format.append([en, ch])

    random.shuffle(corpus_format)
    train_examples, val_examples = corpus_format[:20000], corpus_format[20000:]

    # Write training text files
    with open('train_en.txt', 'w', encoding='utf-8') as f_en, open('train_ch.txt', 'w', encoding='utf-8') as f_ch:
        for en, ch in train_examples:
            f_en.write(en + '\n')
            f_ch.write(ch + '\n')

    # Train SentencePiece models
    vocab_size = 4000  # 或者 4096，接近你的实际 subword 数量上限

    spm.SentencePieceTrainer.Train(f'--input=train_en.txt --model_prefix=en_tokenizer --vocab_size={vocab_size}')
    spm.SentencePieceTrainer.Train(f'--input=train_ch.txt --model_prefix=ch_tokenizer --vocab_size={vocab_size}')

    # Load tokenizers
    tokenizer_en = spm.SentencePieceProcessor(model_file='en_tokenizer.model')
    tokenizer_ch = spm.SentencePieceProcessor(model_file='ch_tokenizer.model')

    sample_str = 'tom intends to play tennis every day during his summer vacation'
    tokenized_str = tokenizer_en.encode(sample_str, out_type=int)
    print(tokenized_str)
    original_str = tokenizer_en.decode(tokenized_str)
    print(original_str)

    def encode(lang):
        lang1, lang2 = lang
        lang1 = [tokenizer_en.vocab_size()] + tokenizer_en.encode(lang1) + [tokenizer_en.vocab_size() + 1]
        lang2 = [tokenizer_ch.vocab_size()] + tokenizer_ch.encode(lang2) + [tokenizer_ch.vocab_size() + 1]
        return [lang1, lang2]

    def filter_long_sent(x, y, max_length=MAX_LEN):
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

    def pad_with_zero(lang, max_length=MAX_LEN):
        lang1, lang2 = lang
        n1 = MAX_LEN - len(lang1)
        n2 = MAX_LEN - len(lang2)
        lang1 = lang1 + [0 for k in range(n1)]
        lang2 = lang2 + [0 for k in range(n2)]
        return [lang1, lang2]
    
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    train_examples, val_examples = corpus_format[:20000], corpus_format[20000:]
    train_examples = [encode(k) for k in train_examples]
    train_examples = [k for k in train_examples if len(k[0]) <= MAX_LEN and len(k[1]) <= MAX_LEN]
    train_examples = [pad_with_zero(k) for k in train_examples]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_examples)

    # 使用缓存数据加速读入
    train_dataset = train_dataset.cache()

    # 打乱并获取批数据
    train_dataset = train_dataset.batch(BATCH_SIZE)

    # 设置预取数据
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # # 验证集数据
    val_examples = [encode(k) for k in val_examples]
    val_examples = [k for k in val_examples if len(k[0]) <= MAX_LEN and len(k[1]) <= MAX_LEN]
    val_examples = [pad_with_zero(k) for k in val_examples]
    val_dataset = tf.data.Dataset.from_tensor_slices(val_examples)

    # 定义超参
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    input_vocab_size = tokenizer_en.vocab_size() + 2
    target_vocab_size = tokenizer_ch.vocab_size() + 2
    max_seq_len = MAX_LEN
    dropout_rate = 0.1

    # 定义优化器
    learing_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9, 
                                        beta_2=0.98, epsilon=1e-9)

    # 定义目标函数和评估指标
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # 创建模型
    transformer = Transformer(num_layers, d_model, num_heads, dff,
                            input_vocab_size, target_vocab_size,
                            max_seq_len, dropout_rate)

    checkpoint_path = './checkpoint/train_ch'
    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)
    # ckpt管理器
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('last checkpoit restore')

    
    EPOCHS = 20
    step_list = []
    loss_list = []
    step = 0

    for epoch in range(EPOCHS):
        start = time.time()

        # 重置记录项
        train_loss.reset_states()
        train_accuracy.reset_states()

        # inputs 英语， targets 汉语

        for batch, all_inputs in enumerate(train_dataset):
            
            # 训练
            inputs = all_inputs[:, 0, :]
            targets = all_inputs[:, 1, :]
            train_step(inputs, targets)

            if batch % 100 == 0:
                loss = train_loss.result()
                print('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
                    epoch+1, batch, loss, train_accuracy.result()
                ))
                step_list.append(step)
                loss_list.append(loss)
            step += 1

        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('epoch {}, save model at {}'.format(
            epoch+1, ckpt_save_path
            ))


        print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
            epoch+1, train_loss.result(), train_accuracy.result()
        ))

        print('time in 1 epoch:{} secs\n'.format(time.time()-start))
    plt.plot(step_list, loss_list)
    plt.xlabel('train step')
    plt.ylabel('loss')


    def evaluate(inp_sentence):
        start_token = [tokenizer_en.vocab_size()]
        end_token = [tokenizer_en.vocab_size() + 1] 
        
        # 输入语句是英语，增加开始和结束标记
        inp_sentence = start_token + tokenizer_en.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # 因为目标是汉语，输入 transformer 的第一个词应该是
        # 汉语的开始标记。
        decoder_input = [tokenizer_ch.vocab_size()]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(MAX_LEN):
            
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input, output)

            predictions, attention_weights = transformer(encoder_input, 
                                                    output,
                                                    False,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)

            # 从 seq_len 维度选择最后一个词
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # 如果 predicted_id 等于结束标记，就返回结果
            if predicted_id == tokenizer_en.vocab_size() + 1: 
                return tf.squeeze(output, axis=0), attention_weights

            # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights
        
    def translate(sentence, plot=''):
        result, attention_weights = evaluate(sentence)
        result_list = result.numpy().tolist()
        predicted_sentence = tokenizer_ch.decode([int(i) for i in result_list if i < tokenizer_ch.vocab_size()])

        print('输入: {}'.format(sentence))
        print('预测输出: {}'.format(predicted_sentence))

    s = corpus_format[20000+7]
    translate(s[0])
    print ("真实输出："+ s[1])
    print("***********")

    s = corpus_format[20000+9]
    translate(s[0])
    print ("真实输出："+ s[1])
    print("***********")

    s = corpus_format[20000+10]
    translate(s[0])
    print ("真实输出："+ s[1])
    print("***********")

    s = corpus_format[20000+12]
    translate(s[0])
    print ("真实输出："+ s[1])