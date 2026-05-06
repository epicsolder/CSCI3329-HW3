# CSCI3329-HW3

Neural Network       mean=0.7288 std=0.0426

=== Final Summary: Mean ± Std (Repeated CV) ===
Linear Classifier     0.6836 ± 0.0530
Logistic Regression   0.7702 ± 0.0402
KNN                   0.7155 ± 0.0430
Gaussian NB           0.7215 ± 0.0443
Neural Network        0.7288 ± 0.0426



Mean accuracy = average performance across 1,000 evaluations

Std (standard deviation) = how much the accuracy fluctuates across folds

Low std → stable model

High std → inconsistent model
first tabol for part 2

=== Final Summary: Mean ± Std (Repeated CV) ===
Linear Classifier     0.6836 ± 0.0530
Logistic Regression   0.7702 ± 0.0402
KNN                   0.7155 ± 0.0430
Gaussian NB           0.7215 ± 0.0443
Neural Network        0.7288 ± 0.0426

=== Part 3: Feature Selection (Forward Selection) ===
the way i foud the aptmol way of the code to use was because of the run ime and mow much the erer form was 

Running forward selection for Linear Classifier...
Best subset for Linear Classifier: [16]
Accuracy: 0.6270

Running forward selection for Logistic Regression...
Best subset for Logistic Regression: [2, 0, 1, 4, 8, 15, 6, 20, 17, 18, 14]
Accuracy: 0.7760

Running forward selection for KNN...
Best subset for KNN: [14]
Accuracy: 0.7000

Running forward selection for Gaussian NB...
Best subset for Gaussian NB: [2, 0, 1, 4, 15, 22]
Accuracy: 0.7625
and for this part the hole proses to find this took 15 min and the other surch took like 30 plus min so i dident let it finish it was taking to long time
Best subset for Neural Network: [2, 0, 1, 3, 5, 17, 8]
Accuracy: 0.7743

Which models improved?
Logistic Regression: 0.7702 → 0.7760

Gaussian NB: 0.7215 → 0.7625

Neural Network: 0.7288 → 0.7743

Which models got worse?
Linear Classifier: 0.6836 → 0.6270

KNN: 0.7155 → 0.7000

These models benefited from removing irrelevant or harmful features.
a
the reson why i chous the fowoerd surch dataset has 24 features. the other one ould have had more rechers to surch 