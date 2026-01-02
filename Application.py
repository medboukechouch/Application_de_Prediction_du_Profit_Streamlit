import streamlit as st
import pandas as pd
from sklearn import impute
import numpy as np
import category_encoders as ce
from sklearn import compose
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

def encodage(x,option):
    if option == "Binary Encoder":
        # Apply Binary Encoder to column 3 (assuming it's categorical)
        ct = compose.ColumnTransformer(
            transformers=[('encoder', ce.BinaryEncoder(), [3])], remainder='passthrough'
        )
        x = ct.fit_transform(x)
        x_new = x  # Binary encoded result
    elif option =="Label Encoder":
        ct = preprocessing.LabelEncoder()
        
        x_state=ct.fit_transform(x[:,3])
        x_state= x_state+1
        x = np.delete(x,3,axis=1)
        x_new =np.column_stack((x_state,x))
        
    else:
        ct = compose.ColumnTransformer(
            transformers=[('encoder', preprocessing.OneHotEncoder(), [3])], remainder="passthrough"
        )
        x = ct.fit_transform(x)
        x_new = x  # OneHot encoded result
        
    return [x,x_new,option]

# Interface pour uploader un fichier CSV
st.title("Application de Prévision du Profit")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    
    
    # Charger les données dans un DataFrame
    df = pd.read_csv(uploaded_file)
    st.subheader("Voila Votre DataFrame : ")
    st.write(df)
    # Affichage des noms des variables independant
    st.subheader("Noms des variables independant")
    x = df.iloc[:,:-1]
    nom_variables_ind = x.columns.tolist()
    st.write(nom_variables_ind)
    
    # Affichage de nom de variable cible
    st.subheader("Nom de variable cible")
    y = df.iloc[:,-1]
    st.write(y.name)

    # Affichage du nombre d’instances et d’attributs
    st.subheader("Nombre d’instances et d’attributs")
    st.write(f"Le dataset contient {df.shape[0]} lignes et {df.shape[1]} colonnes.")

    # Affichage des premières lignes
    st.subheader("Aperçu des données")
    st.dataframe(df.head())
    
    #Affichage des colonnes contenant des valeurs manquantes
    st.subheader("Colonne contenant les valeurs monquants")
    colNull= df.isnull().sum()
    colNull= colNull[colNull>0]
    
    if(colNull.empty):
        st.write("Non colonnes trouvees")
    else:
        st.write(colNull)
        
    st.subheader("Nombre de lignes contenant des valeurs manquantes")
    num_rows_with_missing = df.isnull().any(axis=1).sum()
    
    if num_rows_with_missing == 0:
        st.write("✅ Toutes les lignes sont complètes.")
    else:
        st.write(f"🚨 {num_rows_with_missing} lignes contiennent au moins une valeur manquante.")
        
    st.subheader("Choisir parmi ces option pour remplacer les valeurs manquantes : ")
    
    option = st.selectbox("", ["mean","median","most_frequent"])
    
    imputer = impute.SimpleImputer(missing_values=np.nan,strategy=option)
    x.iloc[:,:-1] = imputer.fit_transform(x.iloc[:,:-1])
    
    st.write(x)
    
    st.subheader("Choisir parmi ces option pour l'encodage  : ")
    
    encod_option = st.selectbox("", ["Binary Encoder","Label Encoder","OneHot Encoder"])
    x = x.to_numpy()
    
    encodage_data = encodage(x,encod_option)
    x = encodage_data [0]
    x_new=encodage_data[1]
    encod_option = encodage_data [2]
    st.write(x_new)
    
    st.subheader("Choisir parmi ces option pour le type de normalisation  : ")
    
    option2 = st.selectbox("",["MaxAbsScaler()","MinMaxScaler()","StandardScaler()","RobustScaler()"])
    
    
    
    
    if encod_option =="Label Encoder":
        sc=eval(f"preprocessing.{option2}")
        scaled = sc.fit_transform(x)
        st.write(scaled)
    else:
        sc=eval(f"preprocessing.{option2}")
        scaled= sc.fit_transform(x[:,-3:])
        st.write(scaled)
        
    corr = df.iloc[:,:-2].corr()
    st.subheader("Matrice de correlation :")
    fig , ax =plt.subplots(figsize=(8,6))
    sns.heatmap(corr,annot=True,cmap='coolwarm' ,fmt='.2f')
    ax.set_title("Matrice de correlation")
    st.pyplot(fig)
    st.subheader("Voulez Vous faire la reduction avec : ")
    option3 = st.radio("",["PCA","Sans reduction"])
    
    if option3 == "PCA":
        st.subheader("Veuillez choisir le nombre de composants : ")
        option2 = st.slider("",min_value=1,max_value=scaled.shape[1],value=2)
        pca = PCA(n_components=option2)
        x_pca = pca.fit_transform(scaled)
        st.write(x_pca)
        
    else:
        x_pca = scaled
        st.write(x_pca)
        
    st.subheader("Choisir pourcentage d'ensemble d'entrainement et de test :")
    option ={
        "90% entraînement - 10% test (📈 utile pour des grands datasets)":0.1,
                 "85% entraînement - 15% test (🏆 pour maximiser l'apprentissage avec suffisamment de tests)":0.15,
                 "80% entraînement - 20% test (💡 le plus courant)":0.2,
                 "70% entraînement - 30% test (✅ bon équilibre)":0.3,
                 "60% entraînement - 40% test (🚨 rare, mais possible si peu de données)":0.4
        }
    
    option2 = st.radio("",list(option.keys()))
    
    train = option[option2]    
    X_train,X_test,Y_train,Y_test=train_test_split(x_pca,y,test_size = train,random_state=42)
    st.write(X_train)
    
    if st.button("Lancer l'entrainement de modele"):
        model = linear_model.LinearRegression()
        model.fit(X_train, Y_train)
        st.session_state.model=model
        st.write("## Le modele est bien traine 😎👌🥶")
        
        
    if st.button("Lancer le modele"):
        if "model" in st.session_state:
            model = st.session_state.model
            Y_pred=model.predict(X_test) 
            comp = pd.DataFrame({'Y_test':Y_test.values.flatten(),'Y_pred':Y_pred.flatten()})
            st.write(comp)
            st.subheader("Les metrics : ")
            st.write('MSE =',metrics.mean_squared_error(Y_test,Y_pred))
            st.write('MAE =',metrics.mean_absolute_error(Y_test,Y_pred))
            st.write('RMSE =',np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
            st.write('R-Squared =',metrics.r2_score(Y_test,Y_pred))
            st.write('Error absolue mediane =',metrics.median_absolute_error(Y_test,Y_pred))
            st.write('Explained variance score =',metrics.explained_variance_score(Y_test,Y_pred))

            plt.figure(figsize=(10, 6))
            plt.plot(Y_test.values.flatten(), label="Valeurs réelles", color="blue", marker="o")
            plt.plot(Y_pred.flatten(), label="Valeurs prédites", color="red", marker="x")
            
            plt.title("Comparaison des valeurs réelles et prédites")
            plt.xlabel("Index")
            plt.ylabel("Valeurs")
            plt.legend()
            
            st.pyplot(plt)
            
            
            
            
    st.subheader("Prédiction avec de nouvelles données")
    new_data = {}
    categorical_columns = ["State"]
    
    for col in nom_variables_ind:
        if col in categorical_columns:
            unique_values = df[col].unique()
            selected_value = st.selectbox(f"Sélectionnez une valeur pour {col}", unique_values)
    
            # Handle encoding based on the selected encoding method
            if encod_option == "OneHot Encoder" or encod_option == "Binary Encoder":
                new_data["State_California"] = 1 if selected_value == "California" else 0
                new_data["State_New York"] = 1 if selected_value == "New York" else 0
                column_order = ["State_California", "State_New York"] + [col for col in nom_variables_ind if col not in categorical_columns]
            elif encod_option == "Label Encoder":
                new_data["State"] = 2 if selected_value == "New York" else 1
                column_order = ["State"] + [col for col in nom_variables_ind if col not in categorical_columns]
                
    
        else:
            # For numerical columns, allow the user to input a value
            new_data[col] = st.number_input(f"Entrez la valeur pour {col}")
    
    # Define the order of columns manually
    
    
    # Create the DataFrame with the specified column order
    new_df = pd.DataFrame([new_data])[column_order]
    
    # Display the ordered new data
    st.write("Ordered New Data:", new_data)
    st.write(new_df)
    
    # Make a prediction
    if st.button("Prédire le profit"):
        if option3 == "PCA":
            # If PCA was applied, transform the new data using the same PCA
            new_df_scaled = sc.fit_transform(new_df.iloc[:,-3:])
            new_df_pca = pca.transform(new_df_scaled)
            st.write(new_df_scaled)
            prediction = st.session_state.model.predict(new_df_pca)
        else:
            # If no PCA was applied, predict directly
            prediction = st.session_state.model.predict(new_df.iloc[:,-3:])
        
        st.write(prediction)


            
            
    


        
