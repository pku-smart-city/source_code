import json
import csv

# 定义文件路径
# path = 'F:\python\workspace\B-GAP with LSTM intention_pre\\rl-agents\scripts\out\HighwayEnv\DQNAgent\dqn_20230322-193654_1524\\'


file_path1 = '0108_train_attack_.csv'

# 写入数据到CSV文件
with open(file_path1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 打开文件,r是读取,encoding是指定编码格式
    for i in range(10):
        episodes = (i+1)*1000
        filename = '/Users/heshouliang/MQLC/rl-agents/scripts/out/HighwayEnv/DQNAgent/MQLC_0410_attack_0714_8000' + '/openaigym.episode_batch.' + str(i) + '.31870.stats.json'

        with open(filename, 'r', encoding='utf-8') as fp:
            # print(type(fp))  # 输出结果是 <class '_io.TextIOWrapper'> 一个文件类对象/home/qiao/he/MQLC/rl-agents/scripts/out/HighwayEnv/DQNAgent/MQLC_0108_train_0110_attack_test4/home/qiao/he/MQLC/rl-agents/scripts/out/HighwayEnv/DQNAgent/MQLC_0111_random_attack_train_random_attack_test1_0211_0

            # load()函数将fp(一个支持.read()的文件类对象，包含一个JSON文档)反序列化为一个Python对象
            data = json.load(fp)

            # print(type(data))  # 输出结果是 <class 'dict'> 一个python对象,json模块会根据文件类对象自动转为最符合的数据类型,所以这里是dict
        print("第", episodes, "个episodes的结果是")


        # 前n个
        n = 90
        max_indices = [i for i, value in sorted(enumerate(data["episode_rewards"]), key=lambda x: x[1], reverse=True)[:n]]
        # print(max_indices)

        a = 0
        b = 0
        c = 0
        for i in max_indices:
            # print(data["episode_speed"][i])
            a += float(data["episode_lengths"][i])
            b += float(sum(data["episode_speed"][i])/len(data["episode_speed"][i]))
            c += float(data["episode_rewards"][i])
        print('lengths:', a/n)
        print('speed:', b/n)
        print('reward:', c/n)

        fp.close()


        # 写入数据到指定行列位置
        writer.writerow([str(a/n)] + [str(b/n)] + [str(c/n)])  # 写入数据行，每行数据前添加一个空格占位