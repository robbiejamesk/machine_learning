import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


class PreProcess:
    def __init__(self):
        self.ds_dataframe = None
        self.unparsed_dataframe = None
        self.pane_dataframe = None
        self.complete_set = None
        self.y = None
        self.x = None

        self.read_and_parse_ds_file()
        self.apply_cell_groupings_for_ds()
        self.read_and_parse_pane_file()

        self.combine_data_frames()
        self.label_encoder()
        self.create_x_y_sets()

    def read_and_parse_ds_file(self):
        imported_csv = pd.read_csv("pane_ds.csv") # read in ds file for pane matched tests
        self.unparsed_dataframe = imported_csv.iloc[1:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 12]]
        column_list = self.unparsed_dataframe.iloc[:,-10:-1].columns
        self.unparsed_dataframe[column_list] = self.unparsed_dataframe[column_list].astype(float)
        self.unparsed_dataframe.CellType = self.unparsed_dataframe.CellType.apply(lambda x: int(str(x)[6:]))

    def apply_cell_groupings_for_ds(self):
        #   df.groupby('A', as_index=False)['B'].agg({'list':(lambda x: list(x))})
        grouped = self.unparsed_dataframe.groupby("CellType")
        df = grouped.aggregate(lambda x: list(x))
        print(df.SampleTemp)
        self.ds_dataframe = self.unparsed_dataframe.groupby("CellType").mean()
        variance_columns = self.unparsed_dataframe.groupby("CellType").var()
        # set variance column titles to before concat
        headers = ['l1Var', 'l2Var', 'l3Var', 'l4Var', 'l5Var', 'l6Var', 'l7Var', 'l8Var', 'SampleTempVar']
        variance_columns.columns = headers
        self.ds_dataframe = pd.concat([variance_columns, self.ds_dataframe], axis=1)

    def read_and_parse_pane_file(self):
        dataframe = pd.read_csv("cell_info.csv")
        self.pane_dataframe = dataframe.iloc[:40, 0:6]

    def combine_data_frames(self):
        self.complete_set = pd.concat([self.pane_dataframe, self.ds_dataframe], axis=1)
        column_list = list(self.complete_set.iloc[:,6:].columns)
        self.complete_set[column_list] = self.complete_set[column_list].shift(-1)
        self.complete_set = self.complete_set.drop(self.complete_set.index[40])
        self.complete_set.iloc[:, 0] = self.complete_set.iloc[:, 0].astype(int)

    def label_encoder(self):
        label_encoder = LabelEncoder()
        self.complete_set['FrontPane'] = label_encoder.fit_transform(self.complete_set['FrontPane'])
        self.complete_set['BackPane'] = label_encoder.fit_transform(self.complete_set['BackPane'])

    def create_x_y_sets(self):
        self.y = self.complete_set.iloc[:, 15:23]
        self.x = self.complete_set.iloc[:, :15]
        self.x = pd.concat([self.x, self.complete_set.iloc[:, 23:]], axis=1)

    def print_values(self):
        print(60 * ('='))
        print("unparsed_dataframe")
        print(self.unparsed_dataframe)
        print(60 * ('='))
        print('ds_dataframe')
        print(self.ds_dataframe)
        print(60 * ('='))
        print("pane_dataframe")
        print(self.pane_dataframe)
        print(60 * ('='))
        print('complete_set')
        print(self.complete_set)
        print(60 * ('='))
        print('x set')
        print(self.x)
        print(60 * ('='))
        print('y set')
        print(self.y)


class Training:

    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

        self.lin_reg = None
        self.test_train_split()
        self.print_values()
        self.linear_regression()

    def test_train_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=0.25, random_state=0)

    def linear_regression(self):
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.x_train.SampleTemp, self.y_train.l1)
        self.y_pred = self.lin_reg.predict(self.x_test.SampleTemp)

        self.regression_scatter()

    def regression_scatter(self):
        # Visualising the Training set results
        plt.scatter(self.x_train.SampleTemp, self.y_train.l1, color='red')
        # plt.plot(self.x_train.SampleTemp, self.y_pred.l1, color='red')
        plt.title('Sample Temp vs DS Output)')
        plt.xlabel('Total Thickness')
        plt.ylabel('DS Output')
        plt.show()

    def print_values(self):
        print(60 * ('='))
        print(" - x - ")
        print(self.x)
        print(60 * ('='))
        print(' - y - ')
        print(self.y)
        print(60 * ('='))

        # # Visualising the Test set results
        # plt.scatter(self.t_test, self.y_test, color='red')
        # plt.plot(X_train, self.lin_reg.predict(self.x_train), color='blue')
        # plt.title('Salary vs Experience (Test set)')
        # plt.xlabel('Years of Experience')
        # plt.ylabel('Salary')
        # plt.show()
        #


pre_processor = PreProcess()
# pre_processor.print_values()
#training = Training(pre_processor.x, pre_processor.y)
#training.print_values()


