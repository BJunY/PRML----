function [knnError, lrError] = test(k)
data = load('D://ex1data1.txt');
X_train = data(1:72, 1);
y_train = data(1:72, 2);
X_test = data(73:97, 1);
y_test = data(73:97, 2);

%compute the Linear Regression Error
lrX_train = [ones(72, 1), X_train];
w = pinv(lrX_train)*y_train;
lrX_test = [ones(25, 1), X_test];
lrError = sqrt(sum((lrX_test * w - y_test).^2));

%compute the KNN Error
knnError = 0;
for i = 1:25
    knnError = knnError + (y_test(i) - KNN(X_train, y_train, k, X_test(i), 1))^2;
end;
knnError = sqrt(knnError);


