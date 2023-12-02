import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig , AutoModelForMaskedLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import json

def load_dataset(file_path, tokenizer):
    # with open(file_path, "r") as f:
    #     data = json.load(f)
    
    data=[
  {
    "keywords": [
      "lipstick",
      "matte",
      "red",
      "long-lasting"
    ],
    "description": "Achieve bold and beautiful lips with our long-lasting matte red lipstick. The intense color payoff ensures a stunning look that lasts all day."
  },
  {
    "keywords": [
      "eyeshadow palette",
      "neutral tones",
      "shimmer",
      "blendable"
    ],
    "description": "Create mesmerizing eye looks with our eyeshadow palette featuring a range of neutral tones. The shimmering shades are highly blendable, allowing you to achieve the perfect eye makeup."
  },
  {
    "keywords": [
      "foundation",
      "full coverage",
      "hydrating",
      "flawless finish"
    ],
    "description": "Experience flawless skin with our hydrating full-coverage foundation. This formula provides a natural and radiant finish, leaving your skin looking perfect all day long."
  },
  {
    "keywords": [
      "mascara",
      "volumizing",
      "lengthening",
      "waterproof"
    ],
    "description": "Get the perfect lashes with our volumizing and lengthening mascara. The waterproof formula ensures a smudge-free look that lasts, giving you long and voluminous lashes."
  },
  {
    "keywords": [
      "blush",
      "rosy",
      "powder",
      "buildable"
    ],
    "description": "Add a touch of rosy glow to your cheeks with our buildable blush powder. The formula is easy to blend, allowing you to achieve a natural and radiant complexion."
  },
  {
    "keywords": [
      "makeup brush set",
      "vegan",
      "cruelty-free",
      "professional"
    ],
    "description": "Upgrade your makeup routine with our professional makeup brush set. Each brush is vegan and cruelty-free, ensuring a flawless application every time."
  },
  {
    "keywords": [
      "highlighter",
      "illuminating",
      "gold",
      "subtle shimmer"
    ],
    "description": "Illuminate your features with our gold-toned highlighter. The illuminating formula adds a subtle shimmer to enhance your natural beauty and create a radiant glow."
  },
  {
    "keywords": [
      "concealer",
      "brightening",
      "full coverage",
      "conceal dark circles"
    ],
    "description": "Conceal imperfections and dark circles with our brightening concealer. The full-coverage formula effortlessly hides blemishes, leaving your skin looking flawless and radiant."
  },
  {
    "keywords": [
      "lip gloss",
      "hydrating",
      "plumping",
      "sheer finish"
    ],
    "description": "Enhance your lips with our hydrating and plumping lip gloss. The sheer finish adds a touch of shine, making your lips look fuller and more defined."
  },
  {
    "keywords": [
      "setting spray",
      "long-lasting",
      "refreshing",
      "makeup setting"
    ],
    "description": "Lock in your makeup look with our long-lasting and refreshing setting spray. This makeup setting spray keeps your makeup in place all day, ensuring a fresh and flawless appearance."
  },
  {
    "keywords": [
      "eyeliner",
      "waterproof",
      "precision",
      "smudge-proof"
    ],
    "description": "Define your eyes with our waterproof and precision eyeliner. The smudge-proof formula ensures a clean and sharp line that lasts, perfect for achieving various eye looks."
  },
  {
    "keywords": [
      "makeup remover",
      "gentle",
      "effective",
      "skin-friendly"
    ],
    "description": "Remove makeup effortlessly with our gentle and effective makeup remover. The skin-friendly formula leaves your skin feeling clean and refreshed without any residue."
  },
  {
    "keywords": [
      "nail polish",
      "vibrant",
      "quick-drying",
      "chip-resistant"
    ],
    "description": "Add a pop of color to your nails with our vibrant nail polish. The quick-drying and chip-resistant formula ensures a long-lasting and flawless manicure."
  },
  {
    "keywords": [
      "makeup bag",
      "stylish",
      "spacious",
      "travel-friendly"
    ],
    "description": "Organize your makeup essentials in our stylish and spacious makeup bag. The travel-friendly design allows you to carry your favorite beauty products wherever you go."
  },
  {
    "keywords": [
      "bronzer",
      "sun-kissed",
      "natural glow",
      "buildable"
    ],
    "description": "Achieve a sun-kissed and natural glow with our buildable bronzer. The formula blends seamlessly, leaving your skin with a radiant and bronzed finish."
  },
  {
    "keywords": [
      "lip liner",
      "definition",
      "long-wearing",
      "precise application"
    ],
    "description": "Define your lips with our long-wearing and precise lip liner. The creamy formula ensures easy application, creating a perfectly contoured pout."
  },
  {
    "keywords": [
      "makeup mirror",
      "LED lights",
      "adjustable",
      "compact"
    ],
    "description": "Illuminate your beauty routine with our makeup mirror featuring adjustable LED lights. The compact design makes it perfect for both home and travel use."
  },
  {
    "keywords": [
      "makeup sponge",
      "latex-free",
      "blending",
      "flawless application"
    ],
    "description": "Achieve a flawless makeup application with our latex-free makeup sponge. The soft and blending texture ensures seamless coverage for a professional finish."
  },
  {
    "keywords": [
      "lip balm",
      "moisturizing",
      "tinted",
      "nourishing"
    ],
    "description": "Nourish and moisturize your lips with our tinted lip balm. The moisturizing formula adds a subtle tint while keeping your lips soft and supple."
  },
  {
    "keywords": [
      "makeup organizer",
      "clear acrylic",
      "multi-compartment",
      "stylish"
    ],
    "description": "Keep your makeup collection organized in our stylish clear acrylic makeup organizer. The multi-compartment design allows you to easily access and display your favorite products."
  },
  {
    "keywords": [
      "makeup primer",
      "pore-filling",
      "smoothing",
      "long-lasting"
    ],
    "description": "Prepare your skin for flawless makeup application with our pore-filling and smoothing makeup primer. The long-lasting formula creates a perfect canvas for a seamless finish."
  },
  {
    "keywords": [
      "makeup brush cleaner",
      "gentle",
      "quick-drying",
      "effective"
    ],
    "description": "Keep your makeup brushes clean with our gentle and effective brush cleaner. The quick-drying formula ensures your brushes are ready for use in no time."
  },
  {
    "keywords": [
      "false eyelashes",
      "dramatic",
      "reusable",
      "easy application"
    ],
    "description": "Enhance your eyes with our dramatic false eyelashes. The reusable lashes are easy to apply, adding instant glamour to your makeup look."
  },
  {
    "keywords": [
      "makeup setting powder",
      "translucent",
      "oil-absorbing",
      "finishing touch"
    ],
    "description": "Complete your makeup look with our translucent setting powder. The oil-absorbing formula provides a flawless finishing touch, ensuring your makeup stays in place all day."
  },
  {
    "keywords": [
      "makeup brush holder",
      "stylish",
      "faux leather",
      "compact"
    ],
    "description": "Organize your makeup brushes in style with our faux leather brush holder. The compact design makes it easy to store and access your favorite brushes, keeping them in pristine condition."
  },
  {
    "keywords": [
      "makeup palette",
      "versatile",
      "travel-friendly",
      "pigmented"
    ],
    "description": "Discover endless possibilities with our versatile and travel-friendly makeup palette. The pigmented shades allow you to create a variety of stunning looks for any occasion."
  },
  {
    "keywords": [
      "setting powder brush",
      "soft bristles",
      "even application",
      "fluffy"
    ],
    "description": "Achieve a flawless finish with our setting powder brush. The soft bristles ensure even application, leaving your makeup set and your skin looking flawless."
  },
  {
    "keywords": [
      "makeup storage",
      "stackable",
      "clear compartments",
      "space-saving"
    ],
    "description": "Maximize your space with our stackable makeup storage solution. Clear compartments make it easy to find your favorite products while keeping your vanity organized."
  },
  {
    "keywords": [
      "liquid eyeliner",
      "precise",
      "water-resistant",
      "intense black"
    ],
    "description": "Define your eyes with our precise liquid eyeliner. The water-resistant formula ensures a long-lasting, intense black line for a bold and dramatic look."
  },
  {
    "keywords": [
      "makeup bag organizer",
      "detachable compartments",
      "portable",
      "easy access"
    ],
    "description": "Stay organized on the go with our makeup bag organizer featuring detachable compartments. The portable design allows for easy access to your beauty essentials wherever you are."
  },
  {
    "keywords": [
      "makeup mirror with stand",
      "adjustable angle",
      "LED lights",
      "elegant design"
    ],
    "description": "Elevate your vanity with our makeup mirror featuring an adjustable angle and LED lights. The elegant design adds a touch of luxury to your beauty routine."
  },
  {
    "keywords": [
      "liquid lipstick",
      "vivid color",
      "smudge-proof",
      "comfortable wear"
    ],
    "description": "Make a statement with our liquid lipstick in vivid colors. The smudge-proof formula ensures a comfortable and long-lasting wear for all-day confidence."
  },
  {
    "keywords": [
      "makeup brush set with bag",
      "travel-sized",
      "essential brushes",
      "premium quality"
    ],
    "description": "Perfect your makeup application on the go with our travel-sized brush set. This set includes essential brushes in a premium-quality bag for beauty on the move."
  },
  {
    "keywords": [
      "compact mirror",
      "dual-sided",
      "magnifying",
      "slim design"
    ],
    "description": "Stay flawless with our compact mirror, featuring dual-sided glass for regular and magnifying views. The slim design fits perfectly in your purse for on-the-go touch-ups."
  },
  {
    "keywords": [
      "makeup sponge set",
      "latex-free",
      "precision",
      "blending"
    ],
    "description": "Achieve a flawless complexion with our latex-free makeup sponge set. The precision and blending capabilities make it a must-have for seamless makeup application."
  },
  {
    "keywords": [
      "makeup brush drying rack",
      "collapsible",
      "quick drying",
      "hygienic"
    ],
    "description": "Keep your brushes in top condition with our collapsible makeup brush drying rack. The quick-drying and hygienic design ensure your brushes are ready for use whenever you need them."
  },
  {
    "keywords": [
      "makeup remover wipes",
      "gentle",
      "refreshing",
      "travel-sized"
    ],
    "description": "Remove makeup effortlessly with our gentle and refreshing makeup remover wipes. The travel-sized pack is perfect for on-the-go convenience."
  },
  {
    "keywords": [
      "makeup artist case",
      "spacious",
      "organized",
      "durable"
    ],
    "description": "Organize your makeup like a pro with our spacious and durable makeup artist case. Stay organized with compartments for brushes, palettes, and all your beauty essentials."
  },
  {
    "keywords": [
      "makeup brush cleaner mat",
      "textured surface",
      "easy cleaning",
      "compact"
    ],
    "description": "Keep your brushes clean with our makeup brush cleaner mat. The textured surface makes cleaning quick and easy, and its compact size is perfect for travel."
  },
  {
    "keywords": [
      "makeup palette organizer",
      "acrylic",
      "clear drawers",
      "chic design"
    ],
    "description": "Display and organize your makeup palettes with our acrylic palette organizer. The clear drawers and chic design add a touch of sophistication to your vanity."
  },
  {
    "keywords": [
      "makeup vanity table",
      "mirror",
      "drawers",
      "stylish"
    ],
    "description": "Transform your beauty routine with our makeup vanity table. Complete with a mirror and drawers, this stylish piece adds elegance to your daily routine."
  },
  {
    "keywords": [
      "makeup brush set with case",
      "professional",
      "synthetic bristles",
      "travel-friendly"
    ],
    "description": "Upgrade your makeup game with our professional brush set featuring synthetic bristles. The travel-friendly case ensures you have your essential tools wherever you go."
  }
]
        
    texts = [item["keywords"] + [""] + [item["description"]] for item in data]
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128  # Adjust the block size as needed
    )

def training():
    # model_name = "allenai/longformer-base-4096"  # You can use other Longformer variants as well
    model_name = "mistralai/Mistral-7B-v0.1"  # You can use other Longformer variants as well
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Replace with your fine-tuning dataset
    train_dataset = load_dataset("product_descriptions.json", tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True  # Now we are doing masked language modeling
    )

    training_args = TrainingArguments(
        output_dir="./fine-tuned-product-model-llm",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Fine-tuning the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine-tuned-product-model-llm")
    tokenizer.save_pretrained("./fine-tuned-product-model-llm")

def generate_product_description(model, tokenizer, keywords):
    input_text = " ".join(keywords) + " "
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate product description
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def testing():
    # Load the fine-tuned model and tokenizer
    model_path = "./fine-tuned-product-model-llm"  # Adjust the path to your fine-tuned Longformer model
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Keywords for testing
    test_keywords = ["lipstick", "vibrant", "long-lasting"]

    # Generate product description
    generated_description = generate_product_description(model, tokenizer, test_keywords)

    # Print the generated description
    print("\nGenerated Product Description:\n")
    print(f"[{generated_description}]\n\n")

if __name__ == "__main__":
    training()
    print(">>Training done!")
    testing()
    print(">>Testing done!")