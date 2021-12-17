# precisions = [1, 0.6666, 0.3, 0.3333, 0.3846, 0.4285, 0.3043]
# recalls = [0.0666, 0.1333, 0.2, 0.2666, 0.333, 0.4, 0.4666]
# tot = len(recalls)  # 总长
# k = 0  # 当前recalls下标
# pre_r = 0  # 记录上一个recall值 初始为0
# ap = 0
# cnt = 1
# while k < tot:
#     max_prec = max(precisions[k:])  # 取后面的最大precision
#     #print("max_prec = ", max_prec)
#     max_idx = precisions[k:].index(max_prec) + k # 取出该precision的下标 实际上也是对应了recalls的下标
#     #print("max_idx = ", max_idx)
#     delta_s = max_prec * (recalls[max_idx] - pre_r)  # 计算面积 高X底
#     ap += delta_s
#     print("A{} = ({:.4f} - {:.4f}) X {:.4f} = {:.8f}".format(cnt, recalls[max_idx], pre_r, max_prec, delta_s))
#     cnt = cnt + 1
#     # print("ap i = ", ap)
#     pre_r = recalls[max_idx]  # 更新pre_r
#     k = max_idx + 1  # k跳到新的位置
#     #if k == tot - 1:  # 到最后一个了就不需要再计算了
#     #    break;
#
# print("AP = {:.2f}%".format(ap*100) )
# print("*"*90)
#
# ap = 0
# for k in range(len(recalls)):
#     maxprec = max(precisions[k:])
#     ap += maxprec
# if (len(recalls) != 0):
#     ap = ap / len(recalls)
# else:
#     ap = 0
# print("AP = {:.2f}%".format(ap*100) )

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') #画图不显示 linux没有图形界面

def plot_confusion_matrix(cm, savename, classes, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    #plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    #plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')

# classes表示不同类别的名称，比如这有6个类别
classes = ['A', 'B', 'C', 'D', 'E', 'F']

random_numbers = np.random.randint(6, size=50)  # 6个类别，随机生成50个样本
y_true = random_numbers.copy()  # 样本实际标签
random_numbers[:10] = np.random.randint(6, size=10)  # 将前10个样本的值进行随机更改
y_pred = random_numbers  # 样本预测标签
np.savetxt("./y_true.txt", y_true,fmt='%d',delimiter=',')
np.savetxt("./y_pred.txt", y_pred,fmt='%d',delimiter=',')
y_true = np.loadtxt("./y_true.txt")
y_pred = np.loadtxt("./y_true.txt")

# 获取混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, 'confusion_matrix.png', classes, title='confusion matrix')