from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import os

if __name__ == "__main__":

    nums_of_data = 1000
    saved_folder = "Results/Generated_From_Template"
    os.makedirs(saved_folder,exist_ok=True)


    # 定义性别，并设置生成比例（8:2）
    genders = ["girl"] * 8 + ["boy"] * 2

    # 定义更加详细的外貌描述（扩展到100种类）
    hair_colors = [
        "blonde hair", "brown hair", "black hair", "red hair", "gray hair", 
        "curly blonde hair", "straight black hair", "short brown hair", "long red hair", "wavy gray hair",
        "bald", "platinum blonde hair", "auburn hair", "silver hair", "chestnut brown hair",
        "dirty blonde hair", "jet black hair", "honey brown hair", "golden blonde hair", "copper red hair",
        "ash brown hair", "strawberry blonde hair", "light brown hair", "dark brown hair", "light black hair",
        "sandy blonde hair", "caramel brown hair", "ginger hair", "pearl white hair", "mahogany hair",
        "espresso brown hair", "charcoal black hair", "bronze brown hair", "rust red hair", "maroon hair",
        "honey blonde hair", "maple brown hair", "butterscotch blonde hair", "cinnamon brown hair", "midnight black hair",
        "light auburn hair", "dark auburn hair", "salt and pepper hair", "plum red hair", "ivory blonde hair",
        "rose gold hair", "cinnamon red hair", "ebony black hair", "dark grey hair", "light grey hair",
        "chocolate brown hair", "cherry red hair", "milky white hair", "carrot orange hair", "hazel brown hair",
        "pecan brown hair", "champagne blonde hair", "mocha brown hair", "sunset orange hair", "ash blonde hair",
        "brunette hair", "sandy brown hair", "mushroom blonde hair", "peach blonde hair", "walnut brown hair",
        "mocha blonde hair", "mulberry red hair", "pearl blonde hair", "pistachio brown hair", "rose brown hair",
        "sand brown hair", "espresso black hair", "nutmeg brown hair", "sunshine blonde hair", "almond brown hair",
        "beige blonde hair", "caramel blonde hair", "cranberry red hair", "honey black hair", "mustard blonde hair",
        "oat blonde hair", "olive brown hair", "pine blonde hair", "raspberry red hair", "saffron blonde hair",
        "sage blonde hair", "slate blonde hair", "teak brown hair", "thistle brown hair", "tobacco brown hair",
        "vanilla blonde hair", "walnut blonde hair", "wine red hair", "almond blonde hair", "hazel blonde hair",
        "clay brown hair", "mango blonde hair", "oyster blonde hair", "pale blonde hair", "russet red hair"
    ]

    eye_colors = [
        "blue eyes", "green eyes", "brown eyes", "hazel eyes", 
        "amber eyes", "gray eyes", "dark brown eyes", "light brown eyes", "pale blue eyes", "forest green eyes",
        "sky blue eyes", "sea green eyes", "chocolate brown eyes", "emerald green eyes", "sapphire blue eyes",
        "copper brown eyes", "deep blue eyes", "aquamarine eyes", "light gray eyes", "midnight blue eyes",
        "mint green eyes", "caramel brown eyes", "cherry brown eyes", "honey brown eyes", "maple brown eyes",
        "mahogany brown eyes", "olive green eyes", "pearl gray eyes", "slate gray eyes", "charcoal gray eyes",
        "bronze brown eyes", "chestnut brown eyes", "cobalt blue eyes", "dark green eyes", "dove gray eyes",
        "hazel green eyes", "ice blue eyes", "indigo blue eyes", "jade green eyes", "khaki green eyes",
        "lavender gray eyes", "lemon yellow eyes", "light green eyes", "lime green eyes", "lilac gray eyes",
        "marble gray eyes", "navy blue eyes", "ocean blue eyes", "olive brown eyes", "onyx black eyes",
        "pale gray eyes", "peach brown eyes", "rose brown eyes", "ruby red eyes", "sand brown eyes",
        "seafoam green eyes", "sepia brown eyes", "smoky gray eyes", "steel blue eyes", "sunflower yellow eyes",
        "tangerine orange eyes", "teal green eyes", "topaz brown eyes", "turquoise blue eyes", "umber brown eyes",
        "vanilla brown eyes", "wine red eyes", "yellow brown eyes", "amber brown eyes", "beige brown eyes",
        "burnt orange eyes", "butterscotch brown eyes", "cinnamon brown eyes", "clay brown eyes", "coal black eyes",
        "coffee brown eyes", "cranberry red eyes", "creamy yellow eyes", "deep green eyes", "dusty gray eyes",
        "ginger brown eyes", "hazel blue eyes", "honey gold eyes", "light brown green eyes", "marigold yellow eyes",
        "milky white eyes", "olive brown green eyes", "peach gray eyes", "pine green eyes", "platinum gray eyes",
        "pumpkin orange eyes", "raisin brown eyes", "rust red eyes", "saffron yellow eyes", "tawny brown eyes"
    ]

    clothing_styles = [
        "wearing a red dress", "in a blue suit", "with a green jacket", "wearing a white t-shirt", "in a yellow raincoat",
        "in a leather jacket", "wearing a black hoodie", "in a floral dress", "wearing jeans and a t-shirt", "in a tracksuit",
        "wearing a business suit", "in a summer dress", "with a winter coat", "in a plaid shirt", "wearing a bomber jacket",
        "in a trench coat", "wearing a tank top", "in a swimsuit", "wearing a kimono", "in a wedding dress",
        "wearing a tuxedo", "in a cardigan", "wearing a sweater", "in a hoodie and jeans", "with a denim jacket",
        "wearing overalls", "in a peacoat", "wearing a scarf", "in a turtleneck", "wearing a vest",
        "in a polo shirt", "wearing a raincoat", "in a bathrobe", "wearing pajamas", "in a blouse and skirt",
        "wearing a sundress", "in a cocktail dress", "wearing a suit and tie", "in a blazer", "wearing cargo pants",
        "in a Hawaiian shirt", "wearing a windbreaker", "in a sports jersey", "wearing hiking boots", "in a parka",
        "wearing a bomber jacket", "in a leather vest", "wearing a jumpsuit", "in a trench coat", "wearing a fur coat",
        "in a silk robe", "wearing a baseball cap", "in a beanie", "wearing gloves", "in a beret",
        "wearing sunglasses", "in a sun hat", "wearing a bow tie", "in a denim skirt", "wearing a maxi dress",
        "in a pencil skirt", "wearing ankle boots", "in high heels", "wearing sandals", "in flip-flops",
        "wearing loafers", "in ballet flats", "wearing cowboy boots", "in a peasant blouse", "wearing a poncho",
        "in a kimono", "wearing a corset", "in a leather skirt", "wearing a satin blouse", "in a sequin dress",
        "wearing a lace dress", "in a chiffon blouse", "wearing a wool coat", "in a cashmere sweater", "wearing a silk scarf",
        "in a velvet blazer", "wearing a tweed jacket", "in a bomber jacket", "wearing a varsity jacket", "in a quilted jacket",
        "wearing a puffer coat", "in a shearling jacket", "wearing a hoodie dress", "in a smocked dress", "wearing a tunic",
        "in a caftan", "wearing a kaftan", "in a duster coat", "wearing a utility jacket", "in a shacket"
    ]

    # 扩展的地点描述（100种类）
    locations = [
        "at Tokyo Tower", "at the beach", "in a park", "on a busy street", "in a quiet library",
        "in a coffee shop", "at a shopping mall", "at the airport", "in a museum", "on a mountain trail",
        "by a river", "in a subway station", "at a concert", "in a sports stadium", "at a carnival",
        "in a classroom", "in a cozy living room", "on a rooftop", "at a wedding ceremony", "in a garden full of flowers",
        "at a construction site", "on a boat", "in a train station", "in a bustling market", "in a quiet alley",
        "on a bridge", "in a small village", "on a farm", "in a factory", "on a highway",
        "in a dense forest", "at a waterfall", "on a frozen lake", "at a desert oasis", "in a jungle",
        "on a snowy mountain", "in a vineyard", "at a lighthouse", "in a castle", "in a palace",
        "in a temple", "at a cathedral", "in a mosque", "in a church", "at a graveyard",
        "in a warzone", "on a battlefield", "in a hospital", "in a laboratory", "in a library",
        "in a bookstore", "at a university", "in a dormitory", "in a hotel", "at a resort",
        "on a tropical island", "at a volcano", "in a rainforest", "in a swamp", "on a dock",
        "at a port", "in a workshop", "at a stadium", "in an arena", "on a soccer field",
        "on a basketball court", "on a tennis court", "on a golf course", "on a race track",
        "in a zoo", "at an aquarium", "in a pet store", "in a toy store", "at a mall food court",
        "at a cinema", "at a theater", "at a science center", "at an art gallery", "at a theme park",
        "in a haunted house", "in a historical monument", "in a war memorial", "at a political rally",
        "in a courtroom", "in a government building", "at a fire station", "at a police station",
        "at a prison", "at a rehabilitation center", "at a circus", "on a ferris wheel", "in a karaoke bar",
        "at a nightclub", "on a beachside boardwalk", "in a mountain cabin", "in a luxury apartment"
    ]

    # 定义动作描述（扩展到100种类）
    actions = [
        "is reading a book", "is taking a selfie", "is running", "is waving", "is eating an ice cream",
        "is drinking coffee", "is painting a picture", "is talking on the phone", "is dancing", "is playing a musical instrument",
        "is cooking", "is fishing", "is swimming", "is riding a bicycle", "is driving a car",
        "is flying a kite", "is playing with a dog", "is building a sandcastle", "is hiking", "is meditating",
        "is doing yoga", "is lifting weights", "is jogging", "is playing soccer", "is playing basketball",
        "is playing tennis", "is playing chess", "is playing video games", "is watching TV", "is watching a movie",
        "is listening to music", "is singing", "is writing a letter", "is drawing", "is sewing",
        "is knitting", "is gardening", "is shopping", "is baking", "is decorating",
        "is cleaning", "is doing laundry", "is feeding birds", "is watering plants", "is grilling",
        "is barbecuing", "is reading a newspaper", "is checking the mail", "is taking out the trash", "is walking a dog",
        "is washing a car", "is riding a motorcycle", "is skateboarding", "is skiing", "is snowboarding",
        "is ice skating", "is surfing", "is playing with children", "is babysitting", "is tutoring",
        "is volunteering", "is donating", "is praying", "is attending a meeting", "is giving a speech",
        "is conducting an interview", "is giving a presentation", "is working on a computer", "is typing",
        "is using a smartphone", "is eating lunch", "is having a picnic", "is waiting for a bus", "is boarding a plane",
        "is checking in at a hotel", "is riding an elevator", "is walking down stairs", "is climbing a ladder", "is opening a door",
        "is closing a window", "is painting a wall", "is hammering a nail", "is sawing wood", "is welding",
        "is fixing a car", "is riding a horse", "is shooting a bow", "is practicing archery", "is playing frisbee",
        "is playing catch", "is playing fetch", "is petting a cat", "is feeding fish", "is riding a rollercoaster",
        "is eating popcorn", "is drinking soda", "is chewing gum", "is reading a map", "is asking for directions"
    ]

    prompt_list = []

    for _ in range(nums_of_data):
        # 随机选择性别、发色、眼睛颜色、衣服风格
        gender = random.choice(genders)
        hair_color = random.choice(hair_colors)
        eye_color = random.choice(eye_colors)
        clothing_style = random.choice(clothing_styles)
        # 组合成详细的外貌描述
        appearance = f"A {gender} with {hair_color} and {eye_color}, {clothing_style}"
        # 随机选择地点和动作
        location = random.choice(locations)
        action = random.choice(actions)
        # 组合成初步提示词
        prompt = f"{appearance} {action} {location}."
        prompt_list.append(prompt)

    # 保存 prompts 到 txt 文件
    with open("{}/prompts.txt".format(saved_folder), "w") as file:
        for prompt in prompt_list:
            file.write(prompt + "\n")

    # 保存各个属性到 txt 文件
    with open("{}/genders.txt".format(saved_folder), "w") as file:
        for gender in genders:
            file.write(gender + "\n")

    with open("{}/hair_colors.txt".format(saved_folder), "w") as file:
        for color in hair_colors:
            file.write(color + "\n")

    with open("{}/eye_colors.txt".format(saved_folder), "w") as file:
        for color in eye_colors:
            file.write(color + "\n")

    with open("{}/clothing_styles.txt".format(saved_folder), "w") as file:
        for style in clothing_styles:
            file.write(style + "\n")

    with open("{}/locations.txt".format(saved_folder), "w") as file:
        for location in locations:
            file.write(location + "\n")

    with open("{}/actions.txt".format(saved_folder), "w") as file:
        for action in actions:
            file.write(action + "\n")
