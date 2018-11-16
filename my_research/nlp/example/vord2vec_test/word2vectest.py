
# 使用gensim word2vec训练脚本获取词向量
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import os.path
import sys
import multiprocessing
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def word2vec_train():
    '''
    word2vec训练
    :return:
    '''

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # inp为输入语料, outp1 为输出模型, outp2为原始c版本word2vec的vector格式的模型
    inp = 'data/射雕英雄传.seg.txt'
    outp1 = 'data/射雕英雄传.model'
    outp2 = 'data/射雕英雄传.vector'
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
    # 保存模型
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

def word2vec_predict(content):
    '''
    Word2vec预测
    :return:
    '''
    try:
        model = gensim.models.Word2Vec.load('data/射雕英雄传.model')
        words = model.most_similar(content) #默认选择前10个
        print('射雕英雄传里面，与"%s"相似的词汇有：' % (content))
        for word in words:
            print(word[0],word[1])
    except:
        print('生僻词语：%s'%content)

def word2vec_predict_wiki(content):
    '''
    加载wiki，Word2vec预测
    :return:
    '''
    model = gensim.models.Word2Vec.load('model/wiki.zh.text.model')
    words = model.most_similar(content) #默认选择前10个
    print('与"%s"相似的词汇有：' % (content))
    for word in words:
        print(word[0],word[1])

if __name__=='__main__':
    # word2vec_train()
    content = '总统'
    word2vec_predict_wiki(content)



