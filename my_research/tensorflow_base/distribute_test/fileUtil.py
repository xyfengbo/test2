
import configparser
def get_config(configkey1,configkey2):
    '''
    读取配置文件
    :param configkey1:
    :param configkey2:
    :param configPath:
    :return:
    '''
    configPath = 'config.txt'
    cf = configparser.ConfigParser()
    cf.read(configPath)
    return cf.get(configkey1,configkey2)