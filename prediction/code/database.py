import sqlite3
import pandas as pd


def saveImage(_id, byte):
    f = open('%s_image.png' % _id, 'wb')
    f.write(byte)
    f.close()
    return


if __name__ == '__main__':
    conn = sqlite3.connect("/Users/lzz/Documents/GitHub/IVF_AI/database/image.pdb")
    cur = conn.cursor()
    cur.execute("select * from IMAGES")

    # 获取table所有字段名
    col_name_list = [tuple[0] for tuple in cur.description]

    # 所有记录列表
    record_all = cur.fetchall()
    df = pd.DataFrame(record_all, columns=col_name_list)

    for index in range(len(df)):
        bytes_id = df['Well'][index]
        bytes_img = df['Image'][index]
        saveImage(bytes_id, bytes_img)

    print("----------------执行完成-----总个数：%s" % len(df))