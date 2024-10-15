# instantiate the custom Model here
base_model = RandomForestClassifierAUC(
    class_weight="balanced",
    max_depth=5,
    random_state=123,
)

# leave this code the same
base_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_model)])

base_pipeline.fit(X_train, y_train)

# calculate the score and compare to the roc_auc_score metric - has it worked?
base_pipeline.score(X_train, y_train), base_pipeline.score(X_test, y_test)
