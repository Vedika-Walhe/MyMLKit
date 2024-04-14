class OneHotEncoder:
    def __init__(self):
        self.dict_unique = {}

    def fit(self, df, columns):
        self.dict_unique = {col: df[col].unique() for col in columns}

    def transform(self, df, columns):
        final_encoded = []
        for col in columns:
            col_encoded = np.zeros((df.shape[0],len(self.dict_unique[col])))
            for i,j in enumerate(df[col]):
                col_idx = np.where(self.dict_unique[col] == j)[0]
                col_encoded[i][col_idx] = 1
            
            final_encoded.append(col_encoded)

        final_encoded = np.concatenate(final_encoded, axis=1)
        return final_encoded
    
    def fit_transform(self, df, columns):
        self.fit(df, columns)
        return self.transform(df, columns)
    

class StandardScaler:
    def fit(self, df, columns):
        self.means = {col: df[col].mean() for col in columns}
        self.std_dev = {col: df[col].std() for col in columns}

    def transform(self, df, columns):
        scaled_array = []
        for col in columns:
            scaled_column = np.zeros((df.shape[0],))
            scaled_column = (df[col] - self.means)/self.std_dev
            scaled_array.append(scaled_column)

        scaled_array = np.concatenate(scaled_array, axis=1)
        return scaled_array

    def fit_transform(self, df, columns):
        self.fit(df, columns)
        self.transform(df, columns)