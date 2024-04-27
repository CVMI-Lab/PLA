import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


model = AutoModelForVision2Seq.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)

prompt = "<grounding>An image of"

url = "https://huggingface.co/ydshieh/kosmos-2-patch14-224/resolve/main/snowman.png"
image = Image.open(requests.get(url, stream=True).raw)

# The original Kosmos-2 demo saves the image first then reload it. For some images, this will give slightly different image input and change the generation outputs.
# Uncomment the following 2 lines if you want to match the original demo's outputs.
# (One example is the `two_dogs.jpg` from the demo)
# image.save("new_image.jpg")
# image = Image.open("new_image.jpg")

inputs = processor(text=prompt, images=image, return_tensors="pt")

generated_ids = model.generate(
    pixel_values=inputs["pixel_values"],
    input_ids=inputs["input_ids"][:, :-1],
    attention_mask=inputs["attention_mask"][:, :-1],
    img_features=None,
    img_attn_mask=inputs["img_attn_mask"][:, :-1],
    use_cache=True,
    max_new_tokens=64,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Specify `cleanup_and_extract=False` in order to see the raw model generation.
processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

print(processed_text)
# `<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.`

# By default, the generated  text is cleanup and the entities are extracted.
processed_text, entities = processor.post_process_generation(generated_text)

print(processed_text)
# `An image of a snowman warming himself by a fire.`

print(entities)
# `[('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]`

