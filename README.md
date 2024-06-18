# linear_regression
从零入手人工智能（3）—— 线性回归
## 1.前言

实践是验证和理解理论知识的重要手段，在进行实际编程之前，我们**首先确保编程环境已正确搭建**。若编程环境尚未搭建完毕，建议参照《从零入手人工智能（2）——搭建开发环境》，文章链接如下：

> https://blog.csdn.net/li_man_man_man/article/details/139537404?spm=1001.2014.3001.5502

线性回归在人工智能中占据重要地位，它通过建立自变量（也称为特征）与因变量（目标变量）之间的线性关系模型，实现对目标变量值的准确预测。该算法因其直观性和计算简便性，成为初学者入门的首选。正如笛卡尔在《方法论》中所提倡的，我们从最基础、最易理解的事务开始，**线性回归便是我们迈出实践的第一步，俗话说实践出真知**。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/1da8cbf3e75b4e9aa305631e77a6c2c4.png#pic_center)

**尽管线性回归在算法层面相对简单，但其应用场景却十分广泛**。例如，在经济预测领域，我们可以利用历史黄金价格数据，通过线性回归模型预测未来的黄金价格走势。同样地，基于中国历年GDP数据，线性回归也能帮助我们预测未来一年的经济增长情况。这些实际应用案例不仅展示了线性回归的强大功能，也体现了其在解决实际问题时的灵活性和实用性。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/238b8ca7a34c4e958c4337ba372d7a4a.png#pic_center)


## 2.线性回归

定义：线性回归是一种统计方法，旨在通过回归分析来确定两个或两个以上**变量之间的依赖关系**，并构建一个线性方程来量化这种关系。
线性回归核心是**确定变量之间的关系**，通过方程表示这种关系。这里提到了变量之间的关系，那么变量之间存在哪些关系呢？

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/261a4ce7a72840a4a89b7b4b0051f023.png#pic_center)

客观世界中普遍存在着变量间的关系,而变量间的关系一般可分为两类:**函数关系、相关关系**。

**函数关系**：存在完全确定的关系，有精确的函数来表示变量间的关系。如圆形的半径r和面积S的关系。
**相关关系**：变量间有着十分密切的关系，但是不能由一个或多个变量值精确的求出另一个变量的值。如身高与体重之间的关系，人的血压与年龄之间的关系。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/7f24358f6d31469fade4281a755182f5.png#pic_center)

相关关系也可分为两种：**平行关系、依存关系**。
**平行关系**：指的是两个或多个元素，它们在逻辑上具有平等或相似性，没有明确的依赖或控制关系。
**依存关系**：指的是两个或多个元素之间的一种非对称关系，其中一个元素对另一个元素具有依赖或控制作用。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/1e0f80d9bc324c8496cc89effb57d349.png#pic_center)


## 3.依赖的工具库

本次的代码依赖了4个工具库：**scikit-learn、pandas、matplotlib、numpy** 。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/4e0719dc187f47738225c665f5365bc9.png#pic_center)


> **Scikit-learn**（也称sklearn）是一个针对Python编程语言的免费软件机器学习库。它提供了各种分类、回归和聚类算法，包含支持向量机、随机森林、梯度提升、k均值和DBSCAN等。
> 
>
> **Matplotlib**是一个Python的2D绘图库，可以绘制各种图形，如折线图、直方图、功率谱、条形图、散点图等。
> 
> **Pandas**是一个基于NumPy的Python数据分析包，提供了高性能的数据结构和数据分析工具。提供了Series（一维数组）和DataFrame（二维表格型数据结构）两种主要的数据结构。支持数据清洗、转换、筛选、排序、分组、聚合等操作。
> 
> **Numpy**是Python的一个开源数值计算扩展，用于存储和处理大型矩阵。提供了N维数组对象（ndarray），支持大量的维度数组与矩阵运算。提供了数学函数库，支持线性代数、傅里叶变换等操作。



## 4.程序框架

本次的代码实现旨在展示线性回归算法的核心功能，其精简版本不超过20行代码，展现了整个实现过程的直观性和高效性。正如谚语“麻雀虽小，五脏俱全”所描述的那样，这个简短的代码片段实则包含了人工智能算法开发的重要**三板斧：数据预处理、模型构建与训练、模型验证**。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/04c1655af05046f6bf12aa04b21b779b.png#pic_center)

**数据预处理阶段**，代码需要能够处理原始数据，可能包括数据的加载、转换、归一化等步骤，以确保输入到模型中的数据是符合算法要求的。
**模型构建与训练阶段**，涉及到了使用线性回归算法建立预测模型，并通过训练数据来优化模型的参数。
**模型验证阶段**，用于评估训练好的模型在未见过的数据上的性能。绘制预测结果与实际结果之间的对比图等可视化手段。

## 5.实战

####  实战一：一元一次线性回归

从简单的入手，我们实战一个一元一次线性回归的程序，在这个程序中有们用到了**三板斧：数据预处理、模型构建与训练、模型验证**。
首先，**手动生成一组X和Y的数据**，这些数据之间存在近似线性关系（Y ≈ 2 * X）。接着，进行数据预处理，使得数据与线性模型输入数据类型一致。

在数据预处理完成后，**构建一个线性回归模型，使用预处理后的X和Y的数据对模型进行训练**，在模型训练完成后，需要对模型进行验证与评估。这一步骤的目的是测试模型的准确性，并判断其是否能够有效拟合数据。通过计算模型的预测值与实际值之间的误差（如均方误差MSE）来评估模型的性能。
**通过验证与评估**，发现模型计算出的线性关系为

> Y = 2.139 * X - 0.0666

这一结果与预设的近似线性关系（Y ≈ 2 * X）基本一致，表明模型具有较高的准确性。

**代码如下：**
```c
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  
  
# 手动生成数据  数据的关系近似 y=2 * x
x = [1,2,3,4,5,6,7,8,9,10]
y = [2,5,8,7,11,13,13,15,18,25]

# 预处理数据 将数据转换成模型匹配的numpy类型
x = np.array(x)
x = x.reshape(-1,1)
y = np.array(y)
y = y.reshape(-1,1)

# 创建线性回归模型  
model = LinearRegression()  
# 训练线性回归模型  
model.fit(x, y)  

# 输出模型及参数  
print('equation：''Y = ', model.coef_[0][0] ,'*X+',model.intercept_[0])  
# 使用模型进行预测 得到预测值y_pred 
predict = model.predict(x) 

mse = mean_squared_error(y, predict)  
r2 = r2_score(y, predict)  
print('MSE:', mse)  
print('R^2 :', r2)  

# 可视化结果   对比 y 和 y_pred
plt.scatter(x, y, color='blue', label='original')  
plt.plot(x, predict, color='red', linewidth=2, label='predict')  
plt.title('linear regression')  
plt.legend()  
plt.show()
```
**代码运行结果如下：**

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ba4b03433d0c40f3864222347009c9e6.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b3a8f1b8baf44f3ebb7f3f3aa0f0f692.png#pic_center)


#### 实战二：一元二次线性回归

难度增加，我们实战一个一元二次线性回归的程序，同样的程序中有们用到了**三板斧：数据预处理、模型构建与训练、模型验证**。
首先，**自动生成一组X和Y的数据，这些数据之间存在近似线性关系**（y ≈  7*x^2 +  3*x  + 5）。接着，我们进行数据预处理，使得数据与线性模型输入数据类型一致。

在数据预处理完成后，**构建一个线性回归模型。使用预处理后的X和Y的数据对模型进行训练**，在模型训练完成后，我们需要对模型进行验证与评估。这一步骤的目的是测试模型的准确性，并判断其是否能够有效拟合数据。通过计算模型的预测值与实际值之间的误差（如均方误差MSE）来评估模型的性能。

**通过验证与评估**，发现模型计算出的线性关系为

> Y =  7.113565229941048 X^2+ 3.037535085400712 X+ 2.2711570860945187。

这一结果与预设的近似线性关系（y ≈  7*x^2 +  3*x  + 5）基本一致，表明模型具有较高的准确性。

**代码如下：**

```c
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.pipeline import make_pipeline   

# 随机生成模拟数据   数据的格式为  numpy类型
np.random.seed(0)  
X = np.linspace(-10, 10, 100).reshape(-1, 1)  
y = 7 * X[:, 0]**2 + 3 * X[:, 0] + 5 +np.random.randn(100) * 10  # 利用随机数制造噪声
# X和Y的关系近似 y = 7*x^2 +  3*x  + 5 

# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)  

# 创建一个管道，其中包括二次多项式特征和线性回归  
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())  
# 训练模型   
model.fit(X_train, y_train)  

# 打印模型的系数（包括截距）  
coef = model.named_steps['linearregression'].coef_  
intercept=  model.named_steps['linearregression'].intercept_
print('equation：''Y = ', coef[2],'*X^2+',coef[1],'*X+',intercept)  

# 利用X_test预测数据得到 y_pred
y_pred = model.predict(X_test)  
# 使用y_test和 y_pred 评估模型 
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  
print('MSE:', mse)  
print('R^2 :', r2)  

# 可视化结果   利用模型使用X生成预测值predict ，对比 y和predict
predict = model.predict(X)  
plt.scatter(X , y, color='blue', label='original')  
plt.plot(X, predict, color='red', linewidth=2, label='predict')  
plt.title('linear regression')  
plt.legend()  
plt.show()
```
**代码运行结果如下：**

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/0d4bfcb535c64162b469ab119287008e.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/e5892c7e1a124009875ec9f8612799c2.png#pic_center)

#### 实战三：二元二次线性回归

继续增加难度，我们实战一个二元二次线性回归的程序，同样的程序中有们用到了**三板斧：数据预处理、模型构建与训练、模型验证**。
首先，**自动生成一组X和Y的数据，这些数据之间存在近似线性关系**

> y ≈  7*x1 + 2X2 + 3X1^2 +9*X1*X2 +4*X2^2 

接着，我们进行数据预处理，使得数据与线性模型输入数据类型一致。

在数据预处理完成后，**构建一个线性回归模型，使用预处理后的X和Y的数据对模型进行训练**，在模型训练完成后，需要对模型进行验证与评估。这一步骤的目的是测试模型的准确性，并判断其是否能够有效拟合数据。通过计算模型的预测值与实际值之间的误差（如均方误差MSE）来评估模型的性能。

**通过验证与评估**，我们发现模型计算出的线性关系为

```c
 Y =  6.962421649874798 *X1+ 2.4917980850336896 *X2+ 3.260715809708717 *X1^2+ 7.685750038820521 *X1*X2+ 3.89457433619066 *X2^2+ 12.031827978992986
```

这一结果与我们预设的近似线性关系（y ≈  7*x1 + 2X2 + 3X1^2 +9*X1*X2 +4*X2^2 ）基本一致，表明模型具有较高的准确性。

**代码如下：**

```c
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression  
from sklearn.pipeline import make_pipeline  
from sklearn.model_selection import train_test_split  
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.metrics import mean_squared_error, r2_score    
# 生成模拟数据  
np.random.seed(0)  
X = np.random.rand(1000, 2)  # 生成1000个随机二维数据样本
X1, X2 = X[:, 0], X[:, 1]   #从X中拆分生成X1 X2
# 设置二次模型系数，按照 [x1, x2, x1^2, x1*x2, x2^2, intercept] 的顺序  
coef = [7, 2, 3, 9, 4, 12]  
# 根据X1 X2 和coef系数生成 y
y = coef[0] * X1 + coef[1] * X2 + coef[2] * X1**2 + coef[3] * X1 * X2 + coef[4] * X2**2 + coef[5]  
y += np.random.randn(1000) * 2  # 添加一些噪声  
  
# 划分训练集和测试集 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)  
# 创建包含多项式特征和线性回归的管道  
# 这里使用degree=2来包含x1, x2, x1^2, x2^2, x1*x2  
#model = PolynomialFeatures(degree=2, include_bias=False)  
# 创建一个管道，其中包括二次多项式特征和线性回归  
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())  
#model = make_pipeline(model, LinearRegression())  
  
# 训练模型  
model.fit(X_train, y_train)  

# 显示模型系数
coef = model.named_steps['linearregression'].coef_  
intercept=  model.named_steps['linearregression'].intercept_
print('equation：''Y = ', coef[0],'*X1+',coef[1],'*X2+',coef[2],'*X1^2+',coef[3],'*X1*X2+',coef[4],'*X2^2+',intercept)  

# 预测  
y_pred = model.predict(X_test)  

# 使用y_test和 y_pred 评估模型 
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  
print('MSE:', mse)  
print('R^2 :', r2)  
# 可视化结果（这里使用3D图来展示）  
x1_plot = np.linspace(X1.min(), X1.max(), 100)[:, np.newaxis]  
x2_plot = np.linspace(X2.min(), X2.max(), 100)[:, np.newaxis]  
x1_plot, x2_plot = np.meshgrid(x1_plot, x2_plot)  
X_plot = np.hstack((x1_plot.ravel()[:, np.newaxis], x2_plot.ravel()[:, np.newaxis]))  
# 预测  
y_plot = model.predict(X_plot)  
  
# 绘制3D曲面  
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='b', marker='o', label='Training Data')  
ax.plot_surface(x1_plot, x2_plot, y_plot.reshape(x1_plot.shape), cmap='viridis', alpha=0.7, label='Fitted Surface')  
ax.set_xlabel('X1')  
ax.set_ylabel('X2')  
ax.set_zlabel('y')  
ax.legend()  
plt.show()
```

**代码运行结果如下：**

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/0caa9dccd07c46fdb25ed69da126c3f8.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/09a153804b1e4b29934fe5e32018ee9e.png#pic_center)


## 6.数据预处理

**在实际应用中，数据的处理往往比模型的选择和训练更为关键**。前面的三个实战案例，数据主要是通过手动或自动方式生成，但实际应用场景中，**数据的加载和预处理步骤至关重要**，这往往是初学者容易忽视的一个环节。

在Scikit-learn中的线性模型对于**输入数据的格式有特定的要求**。对于训练数据X，它通常需要是**二维以及二维以上的数组**，其中每一行代表一个样本，每一列代表一个特征。而对于目标值Y（即标签），它通常是**一维或二维的数组**。

**在实战一（一元一次线性回归）中**，我们手动生成了一个x（特征）和y（目标值）的一维数组，但在将它们输入模型进行训练之前，我们需要对数据的格式进行预处理，需要将一维的x和y转换为二维的NumPy数组（列向量），以满足Scikit-learn线性模型对于输入数据格式的要求。最后我们将转换后的x和y输入模型进行训练。

```c
# 手动生成数据  数据的关系近似 y=2 * x
x = [1,2,3,4,5,6,7,8,9,10]
y = [2,5,8,7,11,13,13,15,18,25]

# 预处理数据 将数据转换成模型匹配的numpy类型
x = np.array(x)
x = x.reshape(-1,1)
y = np.array(y)
y = y.reshape(-1,1)
```
**在实战二（一元二次线性回归）中**，首先我们利用随机数生成器自动创建了一个NumPy类型的二维数组x，基于这个二维数组x，通过定义一元二次函数关系，计算并生成了对应的一维目标值数组y。这一步骤确保了y中的每个值都是基于x中的相应样本通过一元二次函数关系计算得出的。x已经是二维的，因此不需要额外的转换，y是一个一维数组，由于make_pipeline生成的模型支持训练数据y是一维数据，因此我们不需要对y进行转换。最后我们可以将x和y输入模型进行训练。

```c
# 随机生成模拟数据   数据的格式为  numpy类型
np.random.seed(0)  
X = np.linspace(-10, 10, 100).reshape(-1, 1)  
y = 7 * X[:, 0]**2 + 3 * X[:, 0] + 5 +np.random.randn(100) * 10  # 利用随机数制造噪声
```
**在实战三（二元二次线性回归）中**，首先我们利用随机数生成器自动创建了一个NumPy类型的二维数组X，然后将X拆分成两列X1和X2，利用X1和X2通过二次函数关系计算得生成一个一维数据y，然后将x和y输入模型进行训练。最后我们可以将x和y输入模型进行训练。

```c
# 生成模拟数据  
np.random.seed(0)  
X = np.random.rand(1000, 2)  # 生成1000个随机二维数据样本
X1, X2 = X[:, 0], X[:, 1]   #从X中拆分生成X1 X2
# 设置二次模型系数，按照 [x1, x2, x1^2, x1*x2, x2^2, intercept] 的顺序  
coef = [7, 2, 3, 9, 4, 12]  
# 根据X1 X2 和coef系数生成 y
y = coef[0] * X1 + coef[1] * X2 + coef[2] * X1**2 + coef[3] * X1 * X2 + coef[4] * X2**2 + coef[5]  
y += np.random.randn(1000) * 2  # 添加一些噪声 
```
这里需要注意：如果使用LinearRegression()建立模型，这种情况下输入的训练数据x必须一维以上数组和y必须是二维数组。如果使用make_pipeline（）建立模型，这种情况下输入的训练数据x必须二维以上数组和y可以是一维数组。**所以Scikit-learn 线性模型的输入训练数据X是二维以上的数组， Y是一维的数组或者二维的数组**。
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/af7cea42e7a541d498ef828f6ce4575d.png#pic_center)

## 7.进阶实战

在实际工作场景中，我们面对的数据往往来源于真实世界的记录，而非手动或自动生成。那如何把我们记录的数据导入程序呢？进阶实战名为“美国房价预测”，在进阶实战中我们的数据记录在excel表格中，程序利用pandas 工具库读取数据x和y，然后建立和训练模型。在实战开始前，我们先在网上下载一个usa_housing_price的表格文件（可以在github上搜素）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/684f9de5f009428eb3c71cfaa8cd236d.png#pic_center)

以下是详细的代码流程：

> **1.数据加载**：使用Python的pandas库加载usa_housing_price表格文件。
> **2.给X和Y赋值**：从usa_housing_price数据中，从数据中读取Price为目标变量（Y），读取剩余的数据为提取特征（X）。
> **3.可视化数据走势**：绘制每一个特征分量与目标变量之间的散点图，以便更直观地理解数据的走势。
> **4.建立和训练模型**：建立一个线性模型，并使用从表格中读取的数据x和y训练模型。 训练模型：使用训练集数据对选定的模型进行训练。这通常涉及到调整模型的超参数以优化其性能。
> **5.评估模型准确性**：评估模型的预测准确性，常评估指标包括均方误差（MSE）和R平方值（R²）。
> **6.可视化结果**：绘制预测值与实际值的对比图，以直观地展示模型的预测性能。


**代码如下：**

```c
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
#加载usa_housing_price表格数据
data = pd.read_csv("usa_housing_price.csv")
# 从表格中读取多列数据，并将这个多列数据赋值给x
x = data.loc[: ,['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Area Population','size']]
# 从表格中读取Price列数据，并将这个一列数据赋值给y
y = data.loc[: ,'Price']
x.head() # 查看x数据  x是一个五维数据

# x是一个五维数据，我们依次用这5个维度 和 Price 生成一个表格，查看数据走势
fig = plt.figure(figsize = (10,10)) # 定义图形大小
fig1 = plt.subplot(231)  #231表示2行3列的第1幅图
plt.scatter(data.loc[:,'Avg. Area Income'],data.loc[:,'Price'])
plt.title('Price VS Income')

fig2 = plt.subplot(232)  
plt.scatter(data.loc[:,'Avg. Area House Age'],data.loc[:,'Price'])
plt.title('Price VS House Age')

fig3 = plt.subplot(233)  
plt.scatter(data.loc[:,'Avg. Area Number of Rooms'],data.loc[:,'Price'])
plt.title('Price VS Area Number of Rooms')

fig4 = plt.subplot(234)  
plt.scatter(data.loc[:,'Area Population'],data.loc[:,'Price'])
plt.title('Price VS Area Population')

fig5 = plt.subplot(235)  
plt.scatter(data.loc[:,'size'],data.loc[:,'Price'])
plt.title('Price VS size')
plt.show()# 显示5个维度的数据走势

#s建立模型
model = LinearRegression()
#训练模型
model.fit(x ,y)
#根据x预测得到predict
predict= model.predict(x )

#评估模型
mse = mean_squared_error(y,predict) 
r2 = r2_score(y,predict)
print('MSE:', mse)  
print('R^2 :', r2)  

#可视化结果 由于x是一个五维数据，无法直接进行可视化。所以我们利用Y的值和预测值进行画图
fig7 = plt.figure(figsize = (8,5))
plt.scatter(y,predict)
plt.show()#形成的数据越是靠近y = x 这条直线 说明预测结果越准确
```

**代码运行结果如下：**

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/287f0de0cc7f4edcb34c1c040f0119a2.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c8d9b56589b8455ab6e5417ec94a6d43.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/1ce5585ea26f4b5a99f057ad19a8816a.png#pic_center)
