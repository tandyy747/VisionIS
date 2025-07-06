from tensorflow.keras.models import load_model

model = load_model("C:/Users/Administrator/my-react-app/models/mobilenetv2finetune/best_model_mobilenetv2_finetune.h5")

# In kiến trúc mô hình
model.summary()

# In layer chi tiết
for i, layer in enumerate(model.layers):
    print(f"{i}. {layer.name} - {layer.__class__.__name__}")
