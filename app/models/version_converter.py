from tensorflow.keras.models import load_model

# Load the original model (if possible)
original_model = load_model('/home/osama/AlzieDet/App/app/models/Model-2D.keras')

# Save it in the current environment's format
original_model.save('/home/osama/AlzieDet/App/app/models/Compatible-Model-2D.keras', save_format='keras')

