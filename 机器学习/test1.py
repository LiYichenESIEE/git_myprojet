import graphviz
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier

# 加载数据集并训练模型
wine = load_wine()
clf = DecisionTreeClassifier().fit(wine.data, wine.target)

# 导出并显示决策树
dot_data = tree.export_graphviz(clf, feature_names=wine.feature_names, class_names=wine.target_names, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("wine_tree")  # 保存到文件
graph.view()  # 查看图像