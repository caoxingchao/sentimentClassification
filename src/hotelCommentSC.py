# coding: utf-8
import sys
import jieba
from os.path import join

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import Word2Vec
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.classification import LogisticRegression, OneVsRest

reload(sys)
sys.setdefaultencoding("utf-8")


def jieba_cut(line):
    # stop_words = ["", "的", "了", "酒店", "是", "很", "我", "也", "在", "有", "都", "住", "还", "就", "和", "说", "到", "去"]
    # seg_list = list(word for word in jieba.cut(line[0], cut_all=True) if word not in stop_words)
    seg_list = list(word for word in jieba.cut(line[0], cut_all=True) if word != "")
    # seg_list = list(word for word in jieba.cut_for_search(line[0]) if word != "")
    return Row(seg_list)


def tf_idf(words):
    hashing_tf = HashingTF(numFeatures=1000, inputCol="words", outputCol="tf")
    tf = hashing_tf.transform(words)
    tf.cache()
    idf = IDF(minDocFreq=3, inputCol="tf", outputCol="features")
    model = idf.fit(tf)
    idf_res = model.transform(tf)
    return idf_res


def word2vec(words):
    word2Vec = Word2Vec(vectorSize=300, minCount=2, seed=42, inputCol="words", outputCol="features")
    model = word2Vec.fit(words)
    w2v = model.transform(words)
    return w2v


def naive_bayes(training, test):  # 特征向量必须为非负值
    testing = test.select("features")
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    model = nb.fit(training)
    result = model.transform(test)
    accuracy = 1.0 * result.rdd.filter(lambda l: l.label == l.prediction).count() / test.count()
    print "朴素贝叶斯模型的正确率为：", accuracy


def decision_tree(training, test):
    dt = DecisionTreeClassifier(maxDepth=4)
    model = dt.fit(training)
    result = model.transform(test)
    accuracy = 1.0 * result.rdd.filter(lambda l: l.label == l.prediction).count() / test.count()
    print "决策树模型的正确率为：", accuracy


def random_forest(training, test):
    rf = RandomForestClassifier(numTrees=7, maxDepth=4, seed=42)
    model = rf.fit(training)
    result = model.transform(test)
    accuracy = 1.0 * result.rdd.filter(lambda l: l.label == l.prediction).count() / test.count()
    print "随机森林模型的正确率为：" , accuracy


def logistic_regression(training, test):
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    model = lr.fit(training)
    result = model.transform(test)
    accuracy = 1.0 * result.rdd.filter(lambda l: l.label == l.prediction).count() / test.count()
    print "Logistic Regression 模型的正确率为：", accuracy


def one_vs_rest(training, test):
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    ovr = OneVsRest(classifier=lr)
    model = ovr.fit(training)
    result = model.transform(test)
    accuracy = 1.0 * result.rdd.filter(lambda l: l.label == l.prediction).count() / test.count()
    print "OneVsRest 模型的正确率为：", accuracy


def main():
    pos_data = spark.read.text(join(data_path, "hotel-10000-pos.txt")).distinct().rdd\
        .map(jieba_cut).map(lambda l: Row(1.0, l[0])).toDF(["label", "words"])
    neg_data = spark.read.text(join(data_path, "hotel-10000-neg.txt")).distinct().rdd\
        .map(jieba_cut).map(lambda l: Row(0.0, l[0])).toDF(["label", "words"])
    data = pos_data.union(neg_data)

    tf_idf_res = tf_idf(data).select(["label", "features"])
    # wrod2vec_res = word2vec(data).select(["label", "features"])

    res_data = tf_idf_res
    training, test = res_data.randomSplit([0.7, 0.3], seed=0)

    # 朴素贝叶斯，特征向量需要非负，不能使用 word2vec 输出的数据
    naive_bayes(training, test)

    # 决策树
    decision_tree(training, test)

    # 随机森林
    random_forest(training, test)

    # Logistic Regression
    logistic_regression(training, test)

    # OneVsRest
    one_vs_rest(training, test)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        data_path = sys.argv[1]  # 数据目录
        # result_path = sys.argv[2]  # 保存结果目录
    else:
        print "用法：$SPARK_HOME/bin/spark-submit --master spark://master:7077 " \
              "hotelComment.py /data_path/"
        sys.exit()

    spark = SparkSession \
        .builder \
        .appName("Hotel Comment") \
        .getOrCreate()

    main()

    spark.stop()