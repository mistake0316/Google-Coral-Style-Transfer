import numpy as np
import tflite_runtime.interpreter as tflite
import platform
from PIL import Image
import argparse

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_interpreter(model_path):
  interpreter = tflite.Interpreter(
      model_path=model_path,
      experimental_delegates=[
         tflite.load_delegate(EDGETPU_SHARED_LIB,{
        })
      ]
  )
  interpreter.allocate_tensors()
  return interpreter

def channel_mean_std(array, channels=[-2,-3], epsilon=1e-5):
  mean = array.copy()
  for c in channels:
    mean = mean.mean(axis=c, keepdims=True)
  var = array-mean
  for c in channels:
    var = var.mean(axis=c, keepdims=True)

  std = (var+epsilon)**.5
  return mean, std


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A Coral Tpu implmentation for ADAIN style Transform")
  parser.add_argument("-c",
                  "--content_path",
                  type=str,
                  default="./images/content/CHICHI_0.jpg")
  
  parser.add_argument("-s",
                  "--style_path",
                  type=str,
                  default="./images/style/udnie.jpg")

  parser.add_argument("-e",
                  "--encoder_path",
                  type=str,
                  default="./model/dark_Encoder_edgetpu.tflite")
  parser.add_argument("-d",
                  "--decoder_path",
                  type=str,
                  default="./model/dark_Decoder_edgetpu.tflite")
  parser.add_argument("--show", type=bool, default=True)
  args = parser.parse_args()


  encoder_path = args.encoder_path
  decoder_path = args.decoder_path
  
  content_path = args.content_path
  style_path = args.style_path
  



  enc_interpreter = load_interpreter(encoder_path)
  dec_interpreter = load_interpreter(decoder_path)
  
  enc_details = dict()
  enc_details["input"] = enc_interpreter.get_input_details()
  enc_details["output"] = enc_interpreter.get_output_details()
  
  dec_details = dict()
  dec_details["input"] = dec_interpreter.get_input_details()
  dec_details["output"] = dec_interpreter.get_output_details()
  
  
  content_image = Image.open(content_path).convert("RGB").resize(enc_details["input"][0]["shape"][1:2+1])
  content_image = np.array(content_image).astype(np.float32)
  content_image = np.expand_dims(content_image, 0)
  style_image = Image.open(style_path).convert("RGB").resize(enc_details["input"][0]["shape"][1:2+1])
  style_image = np.array(style_image).astype(np.float32)
  style_image = np.expand_dims(style_image, 0)
  
  enc_interpreter.set_tensor(enc_details["input"][0]["index"], style_image)
  enc_interpreter.invoke()
  style_code = enc_interpreter.get_tensor(enc_details["output"][0]["index"])
  style_mean, style_std = channel_mean_std(style_code)
  
  
  enc_interpreter.set_tensor(enc_details["input"][0]["index"], content_image)
  enc_interpreter.invoke()
  content_code = enc_interpreter.get_tensor(enc_details["output"][0]["index"])
  content_mean, content_std = channel_mean_std(content_code)
  
  final_code = (content_code-content_mean)/content_std*style_std + style_mean
  dec_interpreter.set_tensor(dec_details["input"][0]["index"], final_code)
  dec_interpreter.invoke()
  
  result = dec_interpreter.get_tensor(dec_details["output"][0]["index"])
 
  if args.show:
    im = Image.fromarray(result[0].clip(0,255).astype(np.uint8), "RGB")
    im.show()
    im2 = Image.fromarray(content_image[0].astype(np.uint8), "RGB")
    im2.show()
  else:
    pass

