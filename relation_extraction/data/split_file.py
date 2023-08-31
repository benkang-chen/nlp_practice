if __name__ == '__main__':
    with open('./data.json', encoding='utf8') as f:
        datas = f.readlines()
        dev_data = datas[:8000]
        train_data = datas[8000:]
    with open('train_data.json', mode='w', encoding='utf-8') as f:
        f.writelines(train_data)
    with open('dev_data.json', mode='w', encoding='utf-8') as f:
        f.writelines(dev_data)

