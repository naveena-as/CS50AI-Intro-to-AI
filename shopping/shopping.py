import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # transfering data from CSV file to a list of dictionaries, each row represented by a dictionary
    data = []
    with open("shopping.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # lists to store the column names and datatype to be converted to
    int_cols = ["Administrative","Informational","ProductRelated","OperatingSystems","Browser","Region","TrafficType"]
    word_to_int_cols = ["Month", "VisitorType", "Weekend"]
    float_cols = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]

    # mappings required for encoding words to integers for the columns in word_to_int_cols list
    month_mapping = {"Jan":0, "Feb":1, "Mar":2, "Apr":3, "May":4, "June":5, "Jul":6, "Aug":7, "Sep":8, "Oct":9, "Nov":10, "Dec":11}
    weekend_revenue_mapping = {"TRUE":1, "FALSE":0}

    # changing datatype to the desired ones and appending to evidence/value list
    evidence = []
    labels = []
    for row in data:
        row_data = []
        for col in row:
            if col in int_cols:
                row_data.append(int(row[col]))
            elif col in float_cols:
                row_data.append(float(row[col]))
            else:
                # converting Month to int
                if col == word_to_int_cols[0]:
                    row_data.append(month_mapping[row[col]])
                # converting VisitorType to int
                elif col == word_to_int_cols[1]:
                    val = 0
                    if row[col] == "Returning_Visitor": val = 1
                    row_data.append(val)
                # converting Weekend to int
                elif col == word_to_int_cols[2]:
                    row_data.append(weekend_revenue_mapping[row[col]])
                # converting Revenue to int
                else:
                    labels.append(weekend_revenue_mapping[row[col]])
        evidence.append(row_data)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    actual_positive = labels.count(1)
    actual_negative = labels.count(0)
    correct_pred_positive = 0
    correct_pred_negative = 0
    for i in range(len(predictions)):
        # correct prediction
        if labels[i] == predictions[i]:
            # correct positive prediction count for sensitivity
            if predictions[i] == 1:
                correct_pred_positive += 1
            # correct negative prediction count for specificity
            elif predictions[i] == 0:
                correct_pred_negative += 1
    sensitivity = correct_pred_positive / actual_positive
    specificity = correct_pred_negative / actual_negative
    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
