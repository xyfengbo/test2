
import jieba
import jieba.analyse
import jieba.posseg as pseg #引入词性标注接口
import codecs, sys


def jieba_segmentation():
    '''
    结巴分词
    :return:
    '''
    filePath = 'data/射雕英雄传.txt'
    f = codecs.open(filePath,'r',encoding='utf-8')
    target = codecs.open('data/射雕英雄传.seg.txt', 'w', encoding='utf-8')

    lineNum = 1
    line = f.readline()
    while line:
        print('---processing ', lineNum, ' article---')
        seg_list = jieba.cut(line, cut_all=False)
        line_seg = ' '.join(seg_list)
        target.writelines(line_seg)
        lineNum = lineNum + 1
        line = f.readline()

    print('well done.')
    f.close()
    target.close()

if __name__=='__main__':
    jieba_segmentation()

