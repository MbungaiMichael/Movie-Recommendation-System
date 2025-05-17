import pandas as pd
import joblib
import gradio as gr


# Load the trained pipeline
file_directory = r"C:\Users\UltraBook 3.1\PycharmProjects\Movie_rec2\cluster_pipeline_v1.0.0.pkl"
pipeline = joblib.load(file_directory)


def predict_cluster(tag, genres, rating, mean, count):
    # Convert input to DataFrame
    input_data = [{"tag": tag, "genres": genres, "rating": rating, "mean": mean, "count": count}]
    input_df = pd.DataFrame(input_data)

    # Transform and predict using the pipeline
    transformed = pipeline.transform(input_df)
    cluster = transformed['cluster'].iloc[0]

    return {
        "cluster": int(cluster),
        "status": "success"
    }


iface = gr.Interface(fn=predict_cluster,
                     inputs=[
                         gr.Textbox(label="Tag"),
                         gr.Textbox(label="Genres"),
                         gr.Slider(minimum=0, maximum=5, label="Rating"),
                         gr.Slider(minimum=0, maximum=5, label="Mean"),
                         gr.Slider(minimum=0, maximum=10000, label="Count")
                     ],
                     outputs='text',
                     title="Movie Recommendation through Clustering"
                     )

iface.launch()
