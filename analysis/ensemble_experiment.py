# Other random forest code here:
df_train = pd.read_csv("/content/gdrive/My Drive/HW5_Data_Files/zip_train.csv", header=None)
df_test = pd.read_csv("/content/gdrive/My Drive/HW5_Data_Files/zip_test.csv", header=None)
list_1_3 = [1, 3]
list_3_5 = [3, 5]

df_1_3_train = df_train[df_train[0].isin(list_1_3)]
df_1_3_train[0] = df_1_3_train[0].replace([1], -1).replace([3], 1)
df_3_5_train = df_train[df_train[0].isin(list_3_5)]
df_3_5_train[0] = df_3_5_train[0].replace([3], -1).replace([5], 1)

df_1_3_test = df_test[df_test[0].isin(list_1_3)]
df_1_3_test[0] = df_1_3_test[0].replace([1], -1).replace([3], 1)
df_3_5_test = df_test[df_test[0].isin(list_3_5)]
df_3_5_test[0] = df_3_5_test[0].replace([3], -1).replace([5], 1)

X_train_1_3 = df_1_3_train.to_numpy()
y_train_1_3 = X_train_1_3[:, 0]
X_train_1_3 = X_train_1_3[:, 1:]
X_train_3_5 = df_3_5_train.to_numpy()
y_train_3_5 = X_train_3_5[:, 0]
X_train_3_5 = X_train_3_5[:, 1:]

X_test_1_3 = df_1_3_test.to_numpy()
y_test_1_3 = X_test_1_3[:, 0]
X_test_1_3 = X_test_1_3[:, 1:]

X_test_3_5 = df_3_5_test.to_numpy()
y_test_3_5 = X_test_3_5[:, 0]
X_test_3_5 = X_test_3_5[:, 1:]

out_of_bag_error_1_3, test_error_1_3 = random_forest(X_train_1_3, y_train_1_3, X_test_1_3, y_test_1_3, 200, 100)
out_of_bag_error_3_5, test_error_3_5 = random_forest(X_train_3_5, y_train_3_5, X_test_3_5, y_test_3_5, 200, 100)

print("1, 3 OOB Error: ", out_of_bag_error_1_3)
print("1, 3 Test Error: ", test_error_1_3)
print("3, 5 OOB Error: ", out_of_bag_error_3_5)
print("3, 5 Test Error: ", test_error_3_5)

bags = [1, 3, 5, 10, 20, 50, 100, 150, 200]
errors_1_3 = []
errors_3_5 = []
for b in bags:
  errors_1_3.append(random_forest(X_train_1_3, y_train_1_3, X_test_1_3, y_test_1_3, b, 100)[0])
  errors_3_5.append(random_forest(X_train_3_5, y_train_3_5, X_test_3_5, y_test_3_5, b, 100)[0])

print(errors_1_3)
print(errors_3_5)

plt.plot(bags, errors_1_3)
plt.xlabel("# bags")
plt.ylabel("OOB Error")
plt.show()

plt.plot(bags, errors_3_5)
plt.xlabel("# bags")
plt.ylabel("OOB Error")
plt.show()
