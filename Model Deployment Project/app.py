import pickle
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Memuat model yang telah dilatih
def load_model():
    with open('xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Fungsi untuk memproses data input
def preprocess_data(input_data):
  # Membuat objek LabelEncoder untuk mengubah 'Gender' menjadi format numerik
  le = LabelEncoder()
  input_data['Gender'] = le.fit_transform(input_data['Gender'])
  
  # Mengatur OneHotEncoder untuk 'Geography' dengan menentukan kategori secara manual
  # Ini memastikan bahwa semua kategori geografis diwakili, bahkan jika tidak ada dalam data sampel
  onehot_encoder = OneHotEncoder(categories=[['France', 'Germany', 'Spain']], sparse_output=False)
  column_transformer = ColumnTransformer(
      [('one_hot_encoder', onehot_encoder, ['Geography'])],  # Kolom untuk di-transform
      remainder='passthrough'                                # Kolom lainnya dibiarkan apa adanya
  )

  # Melakukan transformasi pada data
  data_transformed = column_transformer.fit_transform(input_data)

  # Membuat daftar nama kolom baru setelah transformasi
  column_names = ['Geography_France', 'Geography_Germany', 'Geography_Spain'] + \
                [col for col in input_data.columns if col != 'Geography']

  # Mengonversi data yang telah ditransformasi menjadi DataFrame
  data_transformed_df = pd.DataFrame(data_transformed, columns=column_names)

  return data_transformed_df

# Antarmuka pengguna Streamlit
def main():
  st.title("Churn Prediction with XGBOOST")
  
  # Input dari pengguna
  credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
  geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
  gender = st.selectbox('Gender', ['Male', 'Female'])
  age = st.number_input('Age', min_value=18, max_value=100, value=30)
  tenure = st.number_input('Tenure', min_value=0, max_value=10, value=2)
  balance = st.number_input('Balance', value=0.0)
  num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
  has_cr_card = st.selectbox('Has Credit Card', [0, 1])
  is_active_member = st.selectbox('Is Active Member', [0, 1])
  estimated_salary = st.number_input('Estimated Salary', value=50000.0)
  
  # Tombol untuk melakukan prediksi
  if st.button('Predict Churn'):
    # Create a DataFrame based on the inputs
    input_dict = {
      'CreditScore': [credit_score],
      'Geography': [geography],
      'Gender': [gender],
      'Age': [age],
      'Tenure': [tenure],
      'Balance': [balance],
      'NumOfProducts': [num_of_products],
      'HasCrCard': [has_cr_card],
      'IsActiveMember': [is_active_member],
      'EstimatedSalary': [estimated_salary]
    }

    # Mengonversi dictionary ke DataFrame
    input_df = pd.DataFrame(input_dict, index=[0])

    # Preprocessing input data
    preprocessed_input = preprocess_data(input_df)
    st.write(preprocessed_input)

    # Prediksi
    prediction = model.predict(preprocessed_input)[0]
    if prediction == 0:
        st.success("0: The customer is likely to stay.")
    else:
        st.error("1: The customer is likely to churn.")
        
if __name__ == '__main__':
  main()
