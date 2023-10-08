def train_classifieres(classifier_data):

    import warnings
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report

    warnings.filterwarnings("ignore")

    converged_elites = classifier_data["converged"]
    not_converged_elites = classifier_data["not_converged"]

    # Combine converged and not converged data
    x = np.vstack((converged_elites, not_converged_elites))
    y = np.hstack((np.ones(len(converged_elites)), np.zeros(len(not_converged_elites))))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(y_train)
    print(y_test)
    percentage_y_0 = (len(y_train[y_train == 0]) / len(y_train)) * 100
    percentage_y_1 = (len(y_train[y_train == 1]) / len(y_train)) * 100
    print(f"Percentage of y_train == 0: {percentage_y_0}")
    print(f"Percentage of y_train == 1: {percentage_y_1}")

    # Initialize classifiers
    classifiers = {
        #"Support Vector Machine": SVC(random_state=1337),
        #"Neural Network": MLPClassifier(random_state=1337),
        "Random Forest": RandomForestClassifier(random_state=1337),
        #"Logistic Regression": LogisticRegression(random_state=1337),
        #"Naive Bayes": GaussianNB(),
        #"K-Nearest Neighbors": KNeighborsClassifier()
    }

    # Train and evaluate classifiers
    results = {}
    for classifier_name, classifier in classifiers.items():
        print(f"Training {classifier_name}...")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        results[classifier_name] = {"Accuracy": accuracy, "Report": report}
        print(f"Accuracy: {accuracy}")
        print(report)

    print(results)

    return results