def adapt_unseen_user(model, support_a, support_b, support_y, test_a, test_b):
    weights=model.adapt_user(support_a,support_b,support_y)
    return model.predict_with_weights(test_a,test_b,weights)
