cfull=DecisionTreeClassifier(random_state=42).fit(x_train,y_train)
path=cfull.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas=path.ccp_alphas
clf_post=max((DecisionTreeClassifier(random_state=42,ccp_alpha=alpha).fit(x_train,y_train) for alpha in ccp_alphas),key=lambda clf:clf.score(x_test,y_test))
print("Accuracy",clf_post.score(x_test,y_test))
plot_tree(clf_post, filled=True, feature_names=iris.feature_names, class_names=list(iris.target_names))
plt.show()
