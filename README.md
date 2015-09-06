# nerve_network
神经网络非线性分类
共有26个类别，分别是1，2，3..26 每条训练数据有617个特征。

训练文件名为train.csv， 内容格式为： id,value0,value1,...,value616,label\n 其中id用于表示每一条训练数据，label表示该条数据的类别，value0表示第0个特征的值，value1表示第一个特征的值，其他依次类推。

测试文件为test.csv， 内容格式为： id,value0,value1,...,value616

预测结果保存于result.csv中， 内容格式为： id,label

训练数据和测试数据可以在这里获得：https://inclass.kaggle.com/c/classification-sysu-2015/data
