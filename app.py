from flask import Flask, jsonify, request, render_template

import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download('punkt', download_dir='/app/nltk_data/')

app = Flask(__name__)
pickled_reviews_data = pickle.load(open('pickles/reviews_data_all_cols.pkl','rb'))
users = list(pickled_reviews_data.reviews_username)
@app.route('/')
def home():
    return render_template('index.html', all_users=users)

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        
        user_input=[str(x) for x in request.form.values()]
        user_input=user_input[0]
        #print(user_input)
        pickled_tfidf_vectorizer = pickle.load(open('pickles/Tfidf_vectorizer_pickle.pkl','rb'))
        pickled_model = pickle.load(open('pickles/Logistic_Reg_final_model.pkl','rb'))
        pickled_user_final_rating = pickle.load(open('pickles/user_final_rating.pkl','rb'))        
        pickled_mapping = pickle.load(open('pickles/prod_id_name_mapping.pkl','rb')) 
        # pickled_reviews_data = pickle.load(open('pickles/reviews_data_all_cols.pkl','rb'))
        # users = list(pickled_reviews_data.reviews_username)

        recommendations = pd.DataFrame(pickled_user_final_rating.loc[user_input]).reset_index()
        recommendations.rename(columns={recommendations.columns[1]: "user_pred_rating" }, inplace = True)
        recommendations = recommendations.sort_values(by='user_pred_rating', ascending=False)[0:20]
       
        recommendations.rename(columns={recommendations.columns[0]: "prod_id" }, inplace = True)
        pickled_mapping.rename(columns={pickled_mapping.columns[0]: "prod_id" }, inplace = True)  
        pickled_reviews_data.rename(columns={pickled_reviews_data.columns[0]: "prod_id" }, inplace = True)
    
        recommendations = pd.merge(recommendations,pickled_mapping, left_on="prod_id", right_on="prod_id", how = "left")
        
        improved_recommendations= pd.merge(recommendations,pickled_reviews_data[['prod_id','reviews_clean']], left_on='prod_id', right_on='prod_id', how = 'left')
        test_data_for_user = pickled_tfidf_vectorizer.transform(improved_recommendations['reviews_clean'].values.astype('U'))
        
        sentiment_prediction_for_user = pickled_model.predict(test_data_for_user)
        sentiment_prediction_for_user = pd.DataFrame(sentiment_prediction_for_user, columns=['Predicted_Sentiment'])

        improved_recommendations= pd.concat([improved_recommendations, sentiment_prediction_for_user], axis=1)
        
        a=improved_recommendations.groupby('prod_id')
        b=pd.DataFrame(a['Predicted_Sentiment'].count()).reset_index()
        b.columns = ['prod_id', 'Total_reviews']        
        c=pd.DataFrame(a['Predicted_Sentiment'].sum()).reset_index()
        c.columns = ['prod_id', 'Total_predicted_positive_reviews']
        
        improved_recommendations_final=pd.merge( b, c, left_on='prod_id', right_on='prod_id', how='left')
        
        improved_recommendations_final['Positive_sentiment_rate'] = improved_recommendations_final['Total_predicted_positive_reviews'].div(improved_recommendations_final['Total_reviews']).replace(np.inf, 0)
        improved_recommendations_final['pos_Recommendation_rate'] = round(improved_recommendations_final['Positive_sentiment_rate']*100,2)
        improved_recommendations_final= improved_recommendations_final.sort_values(by=['Positive_sentiment_rate'], ascending=False )
        improved_recommendations_final=pd.merge(improved_recommendations_final, pickled_mapping, left_on='prod_id', right_on='prod_id', how='left')
        
        name_display= improved_recommendations_final.head(5)
        name_display= name_display[['name','pos_Recommendation_rate']]
        print(name_display)

        output = name_display
        return render_template('index.html', prediction_text=output , all_users = users, user_input  = user_input)
    else :
        return render_template('index.html')
    
if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)