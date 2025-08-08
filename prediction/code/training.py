from config import *
from models import create_model

def train_main_model(X_train, y_train, X_val, y_val, feature_dim, save_model_path):
    """训练主模型"""
    print("=== 训练主模型 ===")
    
    # 多卡支持
    strategy = tf.distribute.MirroredStrategy()
    print(f"可用GPU数量: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = create_model(feature_dim)
        print("模型结构:")
        model.summary()

    # 回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1),
        ModelCheckpoint(save_model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # 数据管道
    batch_size = 32
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 训练模型
    history = model.fit(
        train_ds, validation_data=val_ds, 
        epochs=50, callbacks=callbacks, verbose=1
    )

    # 打印训练结果
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])
    best_val_mae = history.history['val_mae'][best_epoch-1]
    
    print(f"最佳epoch: {best_epoch}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳验证MAE: {best_val_mae:.4f}")
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    print("=== 模型评估 ===")
    
    # 清理GPU内存
    tf.keras.backend.clear_session()
    
    # 分批评估
    batch_size = 16
    test_losses = []
    test_maes = []
    
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch_X = X_test[i:batch_end]
        batch_y = y_test[i:batch_end]
        
        batch_loss, batch_mae = model.evaluate(batch_X, batch_y, verbose=0)
        test_losses.append(batch_loss)
        test_maes.append(batch_mae)
    
    test_loss = np.mean(test_losses)
    test_mae = np.mean(test_maes)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试MAE: {test_mae:.4f}")
    
    return test_loss, test_mae 