# Save the best model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)