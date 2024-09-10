import random
from tqdm import tqdm
import os

if __name__ == "__main__":
    
    nums_of_examples = 1000
    saved_folder = "Results/Generated_From_Template"
    os.makedirs(saved_folder,exist_ok=True)
    
    genders = ["girl"] * 8 + ["boy"] * 2

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
    # 扩展的合理地点和动作组合，包含世界著名景点和更多的活动组合，确保生成1000个唯一的组合
    location_action_pairs = {
        "at the beach": ["is building a sandcastle", "is swimming", "is surfing", "is sunbathing", "is playing volleyball", 
                         "is collecting seashells", "is walking along the shore", "is reading a book", "is flying a kite", "is playing frisbee"],
        "in a park": ["is jogging", "is playing frisbee", "is walking a dog", "is having a picnic", "is riding a bicycle",
                      "is feeding the ducks", "is reading a book", "is playing soccer", "is practicing yoga", "is flying a kite"],
        "in a coffee shop": ["is drinking coffee", "is reading a book", "is working on a laptop", "is chatting with friends", "is sketching",
                             "is writing in a journal", "is studying", "is browsing the internet", "is ordering pastries", "is enjoying a latte"],
        "at the airport": ["is waiting for a flight", "is checking in", "is boarding a plane", "is eating at a restaurant", "is shopping for souvenirs",
                           "is reading a newspaper", "is talking on the phone", "is charging a phone", "is watching planes take off", "is napping in a chair"],
        "in a museum": ["is admiring the exhibits", "is taking photos", "is sketching", "is learning about history", "is listening to an audio guide",
                        "is discussing art with a friend", "is browsing the gift shop", "is attending a lecture", "is watching a documentary", "is exploring ancient artifacts"],
        "on a mountain trail": ["is hiking", "is taking photos", "is enjoying the view", "is having a snack", "is resting by a stream", 
                                "is identifying plants", "is spotting wildlife", "is setting up a tent", "is following a trail map", "is climbing a rock face"],
        "at a concert": ["is dancing", "is singing along", "is cheering", "is enjoying the music", "is taking photos of the band", 
                         "is buying merchandise", "is drinking a beer", "is socializing with friends", "is waiting in line for the restroom", "is clapping after a song"],
        "in a sports stadium": ["is watching a game", "is cheering for the team", "is eating snacks", "is waving a flag", "is buying team merchandise", 
                                "is singing the national anthem", "is high-fiving other fans", "is painting a face with team colors", "is checking the score", "is celebrating a goal"],
        "in a library": ["is reading a book", "is studying", "is writing", "is researching", "is browsing the shelves", 
                         "is using a computer", "is attending a book club", "is checking out books", "is sitting quietly", "is helping a librarian"],
        "at a wedding ceremony": ["is taking photos", "is dancing", "is congratulating the couple", "is enjoying the reception", "is eating cake", 
                                  "is making a toast", "is catching the bouquet", "is listening to speeches", "is socializing with guests", "is watching the first dance"],
        "in a garden": ["is watering plants", "is planting flowers", "is trimming bushes", "is pulling weeds", "is picking vegetables",
                        "is enjoying the view", "is taking photos", "is reading a book", "is having a picnic", "is relaxing in a hammock"],
        "at the gym": ["is lifting weights", "is running on a treadmill", "is doing yoga", "is using a rowing machine", "is attending a fitness class",
                       "is stretching", "is cooling down", "is using the elliptical", "is cycling", "is doing a circuit workout"],
        "in a classroom": ["is listening to a lecture", "is taking notes", "is asking questions", "is doing a group project", "is presenting in front of the class",
                           "is writing on the board", "is reading a textbook", "is discussing with classmates", "is working on an assignment", "is preparing for an exam"],
        "in a hospital": ["is visiting a patient", "is talking to a doctor", "is waiting in the lobby", "is reading a magazine", "is taking medication",
                          "is undergoing a checkup", "is receiving treatment", "is filling out forms", "is speaking with a nurse", "is resting in a bed"],
        "at a party": ["is dancing", "is talking to friends", "is eating snacks", "is playing games", "is drinking punch",
                       "is taking photos", "is laughing", "is singing karaoke", "is making a toast", "is giving a gift"],
        "on a train": ["is reading a newspaper", "is listening to music", "is looking out the window", "is sleeping", "is talking to a fellow passenger",
                       "is working on a laptop", "is having a snack", "is watching a movie", "is browsing the internet", "is writing in a journal"],
        "in a restaurant": ["is ordering food", "is eating a meal", "is talking with a friend", "is enjoying a dessert", "is paying the bill",
                            "is tasting wine", "is taking a photo of the food", "is reading a menu", "is watching the chef", "is listening to live music"],
        "on a bus": ["is reading a book", "is listening to music", "is talking to the driver", "is looking out the window", "is texting on a phone",
                     "is taking a nap", "is playing a game on a phone", "is chatting with a friend", "is waiting for the next stop", "is writing in a notebook"],
        "in a zoo": ["is watching the animals", "is taking photos", "is feeding the animals", "is reading information boards", "is attending a show",
                     "is buying souvenirs", "is enjoying the park", "is having a picnic", "is listening to a guide", "is observing wildlife"],
        "at a mall": ["is shopping for clothes", "is buying gifts", "is browsing stores", "is eating at the food court", "is watching a movie",
                      "is trying on clothes", "is reading a book", "is drinking coffee", "is carrying shopping bags", "is waiting for a friend"],
        "in a cinema": ["is watching a movie", "is eating popcorn", "is buying a ticket", "is talking with friends", "is watching trailers",
                        "is sitting in a seat", "is laughing at a scene", "is enjoying a drink", "is sharing snacks", "is discussing the plot"],
        "at a theater": ["is watching a play", "is applauding the actors", "is buying a ticket", "is reading a program", "is discussing the performance",
                         "is enjoying the show", "is waiting for the intermission", "is sitting in a seat", "is meeting the actors", "is admiring the costumes"],
        "in a city": ["is sightseeing", "is taking photos of landmarks", "is exploring neighborhoods", "is shopping in stores", "is eating at a cafe",
                      "is riding public transport", "is walking through parks", "is visiting museums", "is enjoying street food", "is talking with locals"],
        "on a boat": ["is fishing", "is enjoying the view", "is taking photos", "is steering the boat", "is relaxing on deck", 
                      "is sunbathing", "is watching the sunset", "is reading a book", "is listening to music", "is chatting with friends"],
        "in a forest": ["is hiking", "is camping", "is observing wildlife", "is gathering firewood", "is enjoying the tranquility",
                        "is taking photos", "is identifying plants", "is setting up a tent", "is lighting a campfire", "is roasting marshmallows"],
        "at a carnival": ["is riding a ferris wheel", "is eating cotton candy", "is playing games", "is watching a parade", "is taking photos with clowns",
                          "is shooting targets", "is trying to win prizes", "is riding a carousel", "is watching a magic show", "is enjoying live music"],
        "on a rooftop": ["is stargazing", "is having a barbecue", "is enjoying the view", "is taking photos", "is relaxing on a lounge chair",
                         "is watching the sunset", "is chatting with friends", "is listening to music", "is reading a book", "is sipping a drink"],
        "at the office": ["is working on a computer", "is attending a meeting", "is making a phone call", "is reviewing documents", "is brainstorming ideas",
                          "is chatting with coworkers", "is preparing a presentation", "is writing reports", "is answering emails", "is organizing files"],
        "at a nightclub": ["is dancing", "is ordering drinks", "is socializing with friends", "is listening to the DJ", "is taking photos",
                           "is enjoying the music", "is singing along to songs", "is talking to strangers", "is watching a live performance", "is celebrating a birthday"],
        "on a ski slope": ["is skiing", "is snowboarding", "is enjoying the view", "is taking a break at a lodge", "is having a snowball fight",
                           "is riding a ski lift", "is taking photos of the snow", "is warming up by the fire", "is sipping hot chocolate", "is chatting with friends"],
        "in a spa": ["is getting a massage", "is relaxing in a hot tub", "is enjoying a facial treatment", "is meditating", "is sipping herbal tea",
                     "is sitting in a sauna", "is reading a magazine", "is chatting with friends", "is applying a face mask", "is listening to soothing music"],
        "on a plane": ["is reading a magazine", "is napping", "is talking to a flight attendant", "is watching a movie", "is eating in-flight snacks",
                       "is reading a book", "is looking out the window", "is listening to music", "is writing in a journal", "is chatting with a fellow passenger"],
        "in a subway": ["is reading a newspaper", "is listening to music", "is standing in a crowded car", "is watching performers", "is waiting for the next stop",
                        "is talking to a stranger", "is checking a map", "is texting on a phone", "is watching the scenery", "is chatting with a friend"],
        "at a concert hall": ["is listening to an orchestra", "is applauding the musicians", "is reading the program", "is watching the conductor", "is admiring the acoustics",
                              "is enjoying the music", "is chatting with friends during intermission", "is taking photos of the stage", "is discussing the performance", "is meeting the musicians"],
        "at a protest": ["is holding a sign", "is chanting slogans", "is marching with a crowd", "is listening to a speaker", "is taking photos",
                         "is handing out flyers", "is talking to fellow protesters", "is reading protest signs", "is documenting the event", "is making a speech"],
        "on a beachside boardwalk": ["is walking along the boardwalk", "is eating ice cream", "is shopping for souvenirs", "is taking photos", "is enjoying the sunset",
                                     "is watching street performers", "is riding a bike", "is chatting with friends", "is listening to music", "is relaxing on a bench"],
        "in a garden full of flowers": ["is admiring the flowers", "is taking photos", "is sitting on a bench", "is reading a book", "is sketching the scenery",
                                        "is having a picnic", "is talking with friends", "is enjoying the fragrance", "is watching the bees", "is taking a nap under a tree"],
        "in a small village": ["is exploring the village", "is visiting a local market", "is talking to villagers", "is taking photos of the scenery", "is enjoying a local meal",
                               "is buying handmade crafts", "is chatting with shop owners", "is attending a local festival", "is walking through narrow streets", "is observing daily life"],
        "in a vineyard": ["is tasting wine", "is picking grapes", "is taking a tour", "is enjoying the view", "is taking photos",
                          "is learning about winemaking", "is chatting with the winemaker", "is buying wine", "is enjoying a picnic", "is relaxing under a tree"],
        "in a palace": ["is touring the palace", "is taking photos", "is admiring the architecture", "is learning about history", "is exploring the gardens",
                        "is attending a guided tour", "is listening to stories about royalty", "is observing the decorations", "is buying souvenirs", "is watching a ceremonial event"],
        "at a lighthouse": ["is climbing to the top", "is taking photos", "is enjoying the view", "is learning about the history", "is walking along the shore",
                            "is watching the waves", "is sketching the lighthouse", "is reading about shipwrecks", "is chatting with the lighthouse keeper", "is observing the surrounding area"],
        "in a historical monument": ["is learning about history", "is taking photos", "is reading plaques", "is touring the monument", "is admiring the architecture",
                                     "is listening to a guide", "is discussing history with friends", "is exploring the surroundings", "is taking a guided tour", "is observing the details"],
        "at a waterfall": ["is taking photos", "is hiking", "is enjoying the view", "is picnicking nearby", "is listening to the sound of the water",
                           "is sitting on a rock", "is sketching the scenery", "is dipping feet in the water", "is relaxing in the shade", "is chatting with friends"],
        "in a haunted house": ["is exploring the rooms", "is taking photos", "is listening to ghost stories", "is watching for paranormal activity", "is feeling the chills",
                               "is discussing the history of the house", "is observing strange occurrences", "is listening for noises", "is taking notes", "is watching for shadows"],
        "in a rain forest": ["is hiking", "is observing wildlife", "is taking photos of plants", "is listening to the sounds of the forest", "is crossing a river",
                             "is identifying species", "is setting up a tent", "is sketching the surroundings", "is collecting samples", "is reading a guidebook"],
        "at a fire station": ["is talking to firefighters", "is touring the station", "is learning about fire safety", "is watching a fire drill", "is sitting in a fire truck",
                              "is observing equipment", "is trying on firefighter gear", "is watching a demonstration", "is listening to a safety briefing", "is taking photos of the trucks"],
        "in a police station": ["is reporting a crime", "is talking to officers", "is sitting in the waiting area", "is filing paperwork", "is learning about law enforcement",
                                "is observing police work", "is listening to a briefing", "is touring the station", "is sitting in a patrol car", "is watching surveillance footage"],
        "at a prison": ["is touring the facilities", "is talking to inmates", "is learning about the prison system", "is observing the daily routine", "is visiting the yard",
                       "is discussing rehabilitation", "is watching a prison work program", "is learning about prison history", "is reading about famous inmates", "is observing security measures"],
        "in a government building": ["is attending a meeting", "is discussing policies", "is filing paperwork", "is talking to officials", "is waiting in the lobby",
                                     "is observing government processes", "is taking a tour", "is watching a debate", "is listening to a speech", "is learning about the building's history"],
        "at the Eiffel Tower": ["is taking photos", "is enjoying the view", "is having a picnic", "is reading a guidebook", "is talking to tourists",
                                "is visiting the top level", "is dining at a nearby cafe", "is learning about the tower's history", "is watching the lights at night", "is sketching the structure"],
        "at the Great Wall of China": ["is hiking", "is taking photos", "is learning about history", "is resting on a bench", "is observing the scenery",
                                       "is talking to a guide", "is reading about the wall's construction", "is buying souvenirs", "is enjoying a packed lunch", "is taking in the panoramic views"],
        "at the Statue of Liberty": ["is taking a ferry ride", "is taking photos", "is learning about history", "is admiring the view", "is reading plaques",
                                     "is visiting the museum", "is talking to a park ranger", "is climbing to the crown", "is observing the harbor", "is sketching the statue"],
        "at the Colosseum": ["is exploring the ruins", "is taking photos", "is learning about Roman history", "is sitting on a stone bench", "is admiring the architecture",
                             "is attending a guided tour", "is listening to stories of gladiators", "is buying souvenirs", "is sketching the structure", "is discussing ancient Rome with friends"],
        "at Machu Picchu": ["is hiking the Inca trail", "is taking photos", "is admiring the ancient ruins", "is learning about the Incan civilization", "is resting in the shade",
                            "is talking to a guide", "is reading about the history", "is observing the landscape", "is sketching the ruins", "is enjoying a packed lunch"],
        "at the Sydney Opera House": ["is taking photos", "is watching a performance", "is admiring the architecture", "is sitting by the harbor", "is learning about the history",
                                      "is attending a guided tour", "is dining at a nearby restaurant", "is observing the skyline", "is sketching the structure", "is talking to locals"],
        "at the Taj Mahal": ["is taking photos", "is admiring the marble structure", "is learning about its history", "is sitting in the garden", "is observing the reflection in the pool",
                             "is attending a guided tour", "is reading about the architecture", "is buying souvenirs", "is sketching the structure", "is talking to fellow visitors"],
        "at the Pyramids of Giza": ["is taking photos", "is riding a camel", "is learning about ancient Egypt", "is exploring the pyramids", "is observing the desert landscape",
                                    "is talking to a guide", "is visiting the Sphinx", "is buying souvenirs", "is sketching the pyramids", "is discussing ancient Egypt with friends"],
        "at Times Square": ["is taking photos", "is watching the billboards", "is shopping in stores", "is eating at a restaurant", "is watching street performers",
                            "is talking to tourists", "is buying souvenirs", "is attending a show", "is watching the New Year's Eve ball drop", "is admiring the city lights"],
        "at the Tokyo Tower": ["is taking photos", "is enjoying the view from the observation deck", "is shopping for souvenirs", "is eating at a nearby restaurant", "is learning about the tower's history",
                               "is attending a light show", "is observing the city skyline", "is sketching the tower", "is reading about Tokyo's history", "is talking to tourists"],
        "at Notre-Dame Cathedral": ["is taking photos", "is admiring the architecture", "is attending a mass", "is learning about the cathedral's history", "is exploring the interior",
                                    "is reading about the restoration efforts", "is lighting a candle", "is sketching the stained glass windows", "is buying souvenirs", "is observing the carvings"],
        "at the Golden Gate Bridge": ["is walking across the bridge", "is taking photos", "is enjoying the view", "is learning about the bridge's construction", "is biking along the path",
                                      "is sketching the bridge", "is visiting the visitor center", "is observing the bay", "is talking to tourists", "is watching the fog roll in"],
        "at the Louvre Museum": ["is admiring the art", "is taking photos", "is learning about the exhibits", "is exploring the galleries", "is reading about the paintings",
                                 "is attending a guided tour", "is discussing art with friends", "is sketching the sculptures", "is buying souvenirs", "is observing the architecture"],
        "at the Leaning Tower of Pisa": ["is taking photos", "is posing with the tower", "is learning about its history", "is buying souvenirs", "is sitting in the nearby park",
                                         "is sketching the tower", "is attending a guided tour", "is talking to fellow tourists", "is observing the tower's tilt", "is enjoying a packed lunch"],
        "at Mount Fuji": ["is hiking", "is taking photos", "is admiring the view", "is enjoying a picnic", "is learning about the volcano's history",
                          "is sketching the mountain", "is talking to fellow hikers", "is observing the landscape", "is visiting a shrine", "is discussing the mountain's significance"],
        "at Christ the Redeemer": ["is taking photos", "is admiring the view", "is learning about the statue's history", "is enjoying the surrounding nature", "is sitting on a bench",
                                   "is sketching the statue", "is talking to fellow visitors", "is reading about the construction", "is observing the landscape", "is attending a guided tour"],
    }

    # 生成1000个合理的human activities组合
    unique_activities = list()
    while len(unique_activities) < nums_of_examples:
        location = random.choice(list(location_action_pairs.keys()))
        action = random.choice(location_action_pairs[location])
        activity = f"{action} {location}"
        unique_activities.append(activity)

    # 生成1000个unique的prompts
    prompts = []
    for activity in unique_activities:
        gender = random.choice(genders)
        hair_color = random.choice(hair_colors)
        eye_color = random.choice(eye_colors)
        clothing_style = random.choice(clothing_styles)
        appearance = f"A {gender} with {hair_color} and {eye_color} {clothing_style}"
        prompt = f"{appearance} {activity}."
        prompts.append(prompt)
        

    # 将生成的prompts输出或保存为txt文件
    with open("{}/reasonable_unique_prompts.txt".format(saved_folder), "w") as file:
        for prompt in prompts:
            file.write(prompt + "\n")