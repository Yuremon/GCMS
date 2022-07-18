from sklearn.ensemble import GradientBoostingClassifier
import tools

path_train = "./data/train/"
path_test = "./data/test/"

X_train_origin, y_train = tools.getDataTransformed(path_train + 'database.csv', path_train)
X_test_origin, y_test = tools.getDataTransformed(path_test + 'database.csv', path_test)

clf = GradientBoostingClassifier(n_estimators=1, learning_rate=0.2, max_depth=8, random_state=0).fit(X_train_origin, y_train)
v = clf.score(X_test_origin, y_test)
print(v)
print("\n")