"""Generate 2000 DPO training pairs for list marker learning.

Chosen responses use (*) as list markers.
Rejected responses use - (dash) as list markers.
"""
import json
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Template data: (category, prompt_template, items)
# ---------------------------------------------------------------------------

TEMPLATES = [
    # Cooking
    ("cooking", "What are the main ingredients needed to make a classic {dish}?",
     [("tomatoes", "garlic", "olive oil", "basil", "salt and pepper", "pasta"),
      ("eggs", "butter", "sugar", "flour", "vanilla extract", "baking powder"),
      ("chicken thighs", "soy sauce", "ginger", "garlic", "sesame oil", "green onions"),
      ("beef", "onions", "carrots", "celery", "red wine", "tomato paste"),
      ("lentils", "cumin", "turmeric", "coriander", "onion", "tomatoes"),
      ]),
    ("cooking", "List the steps to prepare {dish}.",
     [("Gather and measure all your ingredients.",
       "Preheat the oven or stovetop to the required temperature.",
       "Prep vegetables by washing, peeling, and chopping as needed.",
       "Combine ingredients according to the recipe order.",
       "Cook for the specified time, stirring or checking as needed.",
       "Taste and adjust seasoning before serving.",
       "Plate the dish and garnish if desired."),
      ("Boil a large pot of salted water.",
       "Cook the pasta until al dente, then drain.",
       "Sauté garlic in olive oil over medium heat.",
       "Add diced tomatoes and simmer for 10 minutes.",
       "Toss pasta in the sauce and add fresh basil.",
       "Serve immediately with grated parmesan."),
      ("Marinate the protein in spices for at least 30 minutes.",
       "Heat oil in a heavy-bottomed pan.",
       "Sear the protein on all sides until browned.",
       "Add aromatics like onion and garlic.",
       "Deglaze with stock or wine.",
       "Simmer on low heat until tender.",
       "Rest before slicing and serving."),
      ]),
    ("cooking", "What are the pros and cons of using an air fryer?",
     [("PRO: Uses significantly less oil than deep frying, reducing calorie intake.",
       "PRO: Cooks food faster than a conventional oven.",
       "PRO: Easy to clean with removable, dishwasher-safe parts.",
       "PRO: Produces a crispy texture similar to deep frying.",
       "CON: Limited capacity — not suitable for large family meals.",
       "CON: Can dry out foods if not monitored carefully.",
       "CON: Takes up significant counter space."),
      ]),

    # Science
    ("science", "What are the key steps in the scientific method?",
     [("Observe a phenomenon and formulate a question.",
       "Research existing knowledge on the topic.",
       "Develop a testable hypothesis.",
       "Design and conduct a controlled experiment.",
       "Collect and analyze the data.",
       "Draw conclusions based on the results.",
       "Communicate findings to the scientific community."),
      ("Define the problem or question clearly.",
       "Gather background information.",
       "Form a hypothesis.",
       "Plan and conduct experiments.",
       "Record and analyze results carefully.",
       "Accept, reject, or modify the hypothesis.",
       "Publish or present the findings."),
      ]),
    ("science", "Name the main layers of the Earth's atmosphere.",
     [("Troposphere — where weather occurs and where we live.",
       "Stratosphere — contains the ozone layer that protects us from UV radiation.",
       "Mesosphere — where most meteors burn up.",
       "Thermosphere — extremely hot layer where auroras occur.",
       "Exosphere — the outermost, thin boundary merging with outer space."),
      ]),
    ("science", "What are the benefits of renewable energy sources?",
     [("Significantly reduces greenhouse gas emissions and air pollution.",
       "Provides energy independence from fossil fuel imports.",
       "Creates jobs in manufacturing, installation, and maintenance.",
       "Offers long-term cost savings once infrastructure is built.",
       "Reduces dependence on finite, non-renewable resources.",
       "Improves public health by reducing air and water pollution.",
       "Powers remote or off-grid communities."),
      ]),
    ("science", "List the planets in our solar system in order from the Sun.",
     [("Mercury — the smallest and closest planet to the Sun.",
       "Venus — the hottest planet due to a thick CO2 atmosphere.",
       "Earth — the only known planet with life.",
       "Mars — the Red Planet with the largest volcano in the solar system.",
       "Jupiter — the largest planet, a gas giant with a great red spot.",
       "Saturn — known for its iconic ring system.",
       "Uranus — an ice giant that rotates on its side.",
       "Neptune — the farthest planet, with the strongest winds."),
      ]),

    # History
    ("history", "What were the main causes of World War I?",
     [("Militarism — European powers had built up large armies and navies.",
       "Alliance systems — interlocking treaties dragged nations into conflict.",
       "Imperialism — competition for colonies created tensions.",
       "Nationalism — ethnic groups sought self-determination.",
       "The assassination of Archduke Franz Ferdinand triggered the war.",
       "A complex web of mobilization orders left little room for diplomacy."),
      ]),
    ("history", "List the major events of the French Revolution.",
     [("1789: Estates-General convened as France faced financial crisis.",
       "July 14, 1789: Storming of the Bastille sparked the revolution.",
       "August 1789: Declaration of the Rights of Man and Citizen adopted.",
       "1791: Constitutional monarchy established.",
       "1792: France declared war on Austria, war spread across Europe.",
       "1793–1794: The Reign of Terror under Robespierre.",
       "1799: Napoleon Bonaparte's coup ended the revolutionary period."),
      ]),
    ("history", "What were the key achievements of the Roman Empire?",
     [("Construction of an extensive road network spanning over 250,000 miles.",
       "Development of advanced aqueduct systems for water supply.",
       "Codification of law that influenced modern legal systems.",
       "Spread of the Latin language across Europe.",
       "Military innovations including legions, engineering, and tactics.",
       "Architectural feats such as the Colosseum and Pantheon.",
       "Trade networks connecting Europe, Africa, and Asia."),
      ]),

    # Technology
    ("technology", "What are the main advantages of cloud computing?",
     [("Scalability — easily increase or decrease resources as needed.",
       "Cost efficiency — pay only for what you use, no upfront hardware costs.",
       "Accessibility — access data and apps from anywhere with internet.",
       "Automatic updates — providers manage software and security patches.",
       "Disaster recovery — data backups and redundancy built in.",
       "Collaboration — teams can work simultaneously on shared documents.",
       "Environmental efficiency — shared infrastructure reduces energy use."),
      ]),
    ("technology", "List the key differences between SQL and NoSQL databases.",
     [("SQL databases use structured, tabular schemas; NoSQL uses flexible, document-based schemas.",
       "SQL enforces ACID transactions; NoSQL often trades consistency for scalability.",
       "SQL scales vertically (bigger servers); NoSQL scales horizontally (more nodes).",
       "SQL is ideal for complex queries and relationships; NoSQL for high-volume, simple queries.",
       "SQL uses a predefined schema; NoSQL schema can evolve over time.",
       "Examples: SQL — MySQL, PostgreSQL; NoSQL — MongoDB, Cassandra."),
      ]),
    ("technology", "What are the steps to set up a basic machine learning pipeline?",
     [("Define the problem and identify the appropriate ML approach.",
       "Collect and curate a representative dataset.",
       "Explore and preprocess the data (handle missing values, normalize, encode).",
       "Split the dataset into training, validation, and test sets.",
       "Select and train a baseline model.",
       "Evaluate performance on the validation set and tune hyperparameters.",
       "Test the final model on the held-out test set.",
       "Deploy the model and monitor for drift or degradation."),
      ]),
    ("technology", "Name the top programming languages used in data science.",
     [("Python — the most popular, with libraries like NumPy, pandas, and scikit-learn.",
       "R — widely used for statistical computing and visualization.",
       "SQL — essential for querying and managing relational databases.",
       "Julia — growing in popularity for high-performance numerical computing.",
       "Scala — used with Apache Spark for big data processing.",
       "MATLAB — common in academic and engineering research."),
      ]),

    # Health
    ("health", "What are the benefits of regular physical exercise?",
     [("Improves cardiovascular health and reduces risk of heart disease.",
       "Helps maintain a healthy body weight and metabolism.",
       "Strengthens muscles and bones, reducing injury risk.",
       "Boosts mental health by releasing endorphins.",
       "Improves sleep quality and duration.",
       "Reduces risk of chronic diseases like type 2 diabetes.",
       "Enhances cognitive function and reduces dementia risk."),
      ]),
    ("health", "List the recommended daily nutrients and their food sources.",
     [("Protein — found in meat, fish, eggs, beans, and lentils.",
       "Carbohydrates — from whole grains, fruits, and vegetables.",
       "Healthy fats — from avocados, nuts, seeds, and olive oil.",
       "Vitamin C — abundant in citrus fruits, strawberries, and bell peppers.",
       "Calcium — in dairy products, leafy greens, and fortified foods.",
       "Iron — in red meat, spinach, and legumes.",
       "Fiber — from whole grains, vegetables, and fruits."),
      ]),
    ("health", "What are the main symptoms of the common cold?",
     [("Runny or stuffy nose — often the first sign of infection.",
       "Sneezing — helps expel the virus from nasal passages.",
       "Sore throat — caused by inflammation from the virus.",
       "Mild fever — the body's immune response to infection.",
       "Coughing — irritation in the throat and airways.",
       "Fatigue and mild body aches — systemic immune response.",
       "Watery eyes — from nasal and sinus inflammation."),
      ]),
    ("health", "What are the pros and cons of a vegetarian diet?",
     [("PRO: Lower risk of heart disease and high blood pressure.",
       "PRO: Reduced environmental impact compared to meat-heavy diets.",
       "PRO: Often associated with lower body weight.",
       "PRO: High in fiber, vitamins, and antioxidants.",
       "CON: Risk of deficiencies in B12, iron, zinc, and omega-3s.",
       "CON: Can be difficult to get complete proteins without careful planning.",
       "CON: May be socially challenging in certain cultural contexts."),
      ]),

    # Travel
    ("travel", "What are the top tourist attractions in {city}?",
     [("The Eiffel Tower — iconic iron lattice tower with panoramic city views.",
       "The Louvre Museum — world's largest art museum and historic monument.",
       "Notre-Dame Cathedral — a masterpiece of French Gothic architecture.",
       "Montmartre — bohemian hilltop neighborhood with Sacré-Cœur Basilica.",
       "The Seine River — take a boat cruise for scenic views of the city.",
       "The Palace of Versailles — opulent royal château just outside the city."),
      ("The Colosseum — ancient amphitheater that held up to 80,000 spectators.",
       "The Vatican Museums and Sistine Chapel — priceless art and history.",
       "The Roman Forum — the heart of ancient Roman civic life.",
       "Trevi Fountain — baroque masterpiece and famous wishing well.",
       "The Pantheon — a remarkably preserved ancient Roman temple.",
       "Piazza Navona — elegant baroque square with fountains and cafés."),
      ("The Sagrada Família — Gaudí's unfinished masterpiece basilica.",
       "Park Güell — colorful mosaic park with city views.",
       "La Boqueria Market — vibrant covered food market.",
       "The Gothic Quarter — medieval streets and historic architecture.",
       "Camp Nou — home of FC Barcelona.",
       "Montjuïc — castle and gardens overlooking the city and sea."),
      ]),
    ("travel", "What should you pack for a two-week trip to {destination}?",
     [("Lightweight, moisture-wicking clothing suitable for the climate.",
       "Comfortable walking shoes with good arch support.",
       "Travel adapter and portable charger.",
       "Photocopies of important documents (passport, insurance, itinerary).",
       "Basic first-aid kit with pain relievers, bandages, and medication.",
       "Reusable water bottle and a day pack.",
       "Camera or extra smartphone storage for photos.",
       "Local currency and a travel credit card with no foreign fees."),
      ]),
    ("travel", "What are the best tips for traveling on a budget?",
     [("Travel during the shoulder season for lower prices and fewer crowds.",
       "Book flights and accommodation well in advance for early-bird deals.",
       "Use public transportation instead of taxis or rental cars.",
       "Stay in hostels, guesthouses, or use home-sharing platforms.",
       "Cook some of your own meals instead of eating out every day.",
       "Look for free attractions like parks, museums with free days, and walking tours.",
       "Use cashback or travel rewards credit cards wisely.",
       "Download offline maps and translation apps to avoid roaming charges."),
      ]),

    # Education
    ("education", "What are the most effective study techniques for exams?",
     [("Active recall — test yourself on material rather than just re-reading.",
       "Spaced repetition — review information at increasing intervals over time.",
       "The Pomodoro technique — 25-minute focused sessions with short breaks.",
       "Mind mapping — visually organize concepts and their connections.",
       "Teaching others — explaining concepts solidifies your own understanding.",
       "Practice past exams — familiarize yourself with question formats.",
       "Interleaving — alternate between subjects rather than blocking one subject."),
      ]),
    ("education", "List the key benefits of bilingual education.",
     [("Improved cognitive flexibility and problem-solving abilities.",
       "Enhanced executive function and multitasking skills.",
       "Greater cultural awareness and empathy.",
       "Better career prospects in an increasingly globalized world.",
       "Delayed onset of age-related cognitive decline.",
       "Stronger metalinguistic awareness — understanding language itself.",
       "Access to a wider range of literature, media, and communities."),
      ]),
    ("education", "What are the main learning styles and their characteristics?",
     [("Visual learners — prefer diagrams, charts, and spatial understanding.",
       "Auditory learners — learn best through listening and discussion.",
       "Reading/writing learners — prefer to process information through text.",
       "Kinesthetic learners — learn by doing, hands-on activities and experiments.",
       "Social learners — thrive in group settings and collaborative projects.",
       "Solitary learners — prefer to study alone and self-reflect."),
      ]),

    # Business
    ("business", "What are the key components of a successful business plan?",
     [("Executive summary — a concise overview of the business and its goals.",
       "Company description — mission, vision, and the problem being solved.",
       "Market analysis — target audience, market size, and competition.",
       "Organization and management — structure and team qualifications.",
       "Product or service description — what you're offering and its value.",
       "Marketing and sales strategy — how you'll attract and retain customers.",
       "Financial projections — revenue model, costs, and profitability timeline.",
       "Funding requirements — how much capital is needed and how it will be used."),
      ]),
    ("business", "List the most important digital marketing channels.",
     [("Search Engine Optimization (SEO) — drives organic traffic from search engines.",
       "Pay-Per-Click (PPC) advertising — paid search and display ads for quick visibility.",
       "Social media marketing — engage audiences on platforms like Instagram and LinkedIn.",
       "Email marketing — direct, personalized communication with subscribers.",
       "Content marketing — blogs, videos, and infographics to attract and educate.",
       "Affiliate marketing — partner with others to promote your products.",
       "Influencer marketing — leverage trusted voices to reach new audiences."),
      ]),
    ("business", "What are the pros and cons of remote work?",
     [("PRO: Eliminates commuting time and costs.",
       "PRO: Greater flexibility in working hours and location.",
       "PRO: Access to global talent pools for employers.",
       "PRO: Often leads to higher employee satisfaction and retention.",
       "CON: Can lead to feelings of isolation and disconnection.",
       "CON: Harder to collaborate spontaneously and build team culture.",
       "CON: Home environments may have distractions and poor ergonomics.",
       "CON: Blurred work-life boundaries can lead to overwork."),
      ]),

    # Environment
    ("environment", "What are the main causes of climate change?",
     [("Burning fossil fuels releases CO2 and other greenhouse gases.",
       "Deforestation reduces Earth's capacity to absorb carbon.",
       "Industrial agriculture produces methane from livestock and fertilizers.",
       "Manufacturing and industrial processes emit nitrous oxide and fluorinated gases.",
       "Transportation — cars, planes, and ships burn fossil fuels.",
       "Waste decomposition in landfills produces methane.",
       "Land use changes disrupt natural carbon sinks."),
      ]),
    ("environment", "List practical ways individuals can reduce their carbon footprint.",
     [("Switch to renewable energy at home (solar panels, green energy tariffs).",
       "Reduce meat and dairy consumption — especially beef.",
       "Use public transport, cycle, or walk instead of driving.",
       "Fly less — consider trains or video calls for meetings.",
       "Buy less and choose second-hand or sustainably sourced products.",
       "Improve home insulation and use energy-efficient appliances.",
       "Reduce food waste by planning meals and composting scraps.",
       "Support businesses and politicians committed to sustainability."),
      ]),

    # Psychology
    ("psychology", "What are the key principles of cognitive behavioral therapy (CBT)?",
     [("Thoughts, feelings, and behaviors are interconnected.",
       "Negative automatic thoughts can be identified and challenged.",
       "Cognitive restructuring replaces distorted thinking with balanced thoughts.",
       "Behavioral activation encourages engagement with positive activities.",
       "Exposure therapy gradually reduces avoidance of feared situations.",
       "Skills are practiced between sessions as homework.",
       "Goals are collaborative, specific, and measurable."),
      ]),
    ("psychology", "List the main stages of Maslow's hierarchy of needs.",
     [("Physiological needs — food, water, shelter, sleep, and clothing.",
       "Safety needs — personal security, employment, health, and property.",
       "Love and belonging needs — friendship, intimacy, family, and community.",
       "Esteem needs — self-esteem, recognition, status, and respect.",
       "Self-actualization — achieving one's full potential and creative fulfillment."),
      ]),

    # Sports
    ("sports", "What are the main benefits of playing team sports?",
     [("Develops teamwork and collaboration skills.",
       "Builds communication and leadership abilities.",
       "Improves physical fitness and coordination.",
       "Teaches discipline, goal-setting, and perseverance.",
       "Provides social connection and a sense of belonging.",
       "Builds resilience through wins and losses.",
       "Can reduce stress and improve mental well-being."),
      ]),
    ("sports", "List the equipment needed to play {sport}.",
     [("Cleats — specialized shoes for grip on grass or turf.",
       "Shin guards — protect lower legs from kicks and impacts.",
       "Jersey and shorts — lightweight, breathable athletic wear.",
       "Socks — long socks to hold shin guards in place.",
       "Soccer ball — size 5 for adults.",
       "Goalkeeper gloves — for the goalkeeper position.",
       "Water bottle — stay hydrated throughout the match."),
      ("Racket — strung with appropriate tension for your play style.",
       "Tennis balls — pressurized balls for regular play.",
       "Athletic shoes — court shoes with lateral support.",
       "Comfortable athletic wear — shorts, t-shirt, or dress.",
       "Wristbands and grip tape — for sweat control.",
       "Ball hopper — for solo practice sessions."),
      ]),

    # Finance
    ("finance", "What are the key principles of personal finance?",
     [("Live below your means — spend less than you earn.",
       "Build an emergency fund covering 3-6 months of expenses.",
       "Pay off high-interest debt as quickly as possible.",
       "Invest early and consistently to benefit from compound interest.",
       "Diversify investments across asset classes to manage risk.",
       "Understand and plan for taxes.",
       "Protect yourself with appropriate insurance coverage.",
       "Review and adjust your financial plan regularly."),
      ]),
    ("finance", "List the main types of investment vehicles.",
     [("Stocks — ownership shares in publicly traded companies.",
       "Bonds — debt securities issued by governments or corporations.",
       "Mutual funds — pooled investment vehicles managed by professionals.",
       "Exchange-traded funds (ETFs) — diversified funds traded like stocks.",
       "Real estate — property investments for rental income or appreciation.",
       "Certificates of deposit (CDs) — fixed-term, FDIC-insured bank deposits.",
       "Commodities — physical goods like gold, oil, and agricultural products."),
      ]),

    # Animals & Nature
    ("nature", "What are the characteristics that define mammals?",
     [("Warm-blooded — maintain a constant internal body temperature.",
       "Have hair or fur covering their bodies.",
       "Give birth to live young (with few exceptions).",
       "Nurse offspring with milk produced by mammary glands.",
       "Have a four-chambered heart.",
       "Breathe air using lungs.",
       "Have a more complex brain compared to other vertebrates."),
      ]),
    ("nature", "List the main biomes found on Earth.",
     [("Tropical rainforest — warm, wet, and highly biodiverse.",
       "Savanna — tropical grassland with scattered trees and distinct wet/dry seasons.",
       "Desert — extremely dry, with extreme temperature variation.",
       "Temperate forest — deciduous trees and four distinct seasons.",
       "Boreal forest (taiga) — coniferous forests of the northern hemisphere.",
       "Tundra — cold, treeless plains with permafrost.",
       "Ocean — the largest biome, covering over 70% of Earth's surface.",
       "Freshwater — rivers, lakes, and wetlands."),
      ]),

    # Music
    ("music", "What are the main elements of music?",
     [("Rhythm — the pattern of beats and timing in music.",
       "Melody — a sequence of notes that forms a recognizable tune.",
       "Harmony — the combination of notes played simultaneously.",
       "Dynamics — the variation in loudness from soft to loud.",
       "Timbre — the unique tone quality or color of an instrument or voice.",
       "Tempo — the speed at which the music is played.",
       "Form — the structure or organization of a musical piece."),
      ]),
    ("music", "List the steps to learn a new instrument as a beginner.",
     [("Choose the right instrument based on your interests and goals.",
       "Find a good teacher or quality online learning resources.",
       "Learn to read music notation or tablature for your instrument.",
       "Practice fundamental techniques daily — scales, chords, or basic exercises.",
       "Start with simple songs you enjoy to stay motivated.",
       "Record yourself to identify areas for improvement.",
       "Play with others or join an ensemble as soon as possible.",
       "Be patient — progress comes with consistent, deliberate practice."),
      ]),

    # Literature
    ("literature", "What are the main literary devices used in poetry?",
     [("Metaphor — a direct comparison between two unlike things.",
       "Simile — a comparison using 'like' or 'as'.",
       "Alliteration — repetition of consonant sounds at the start of words.",
       "Assonance — repetition of vowel sounds within words.",
       "Personification — giving human qualities to non-human things.",
       "Imagery — vivid descriptive language appealing to the senses.",
       "Symbolism — using one thing to represent a deeper meaning.",
       "Rhyme — similarity of sound at the end of lines."),
      ]),

    # Architecture
    ("architecture", "What are the defining features of Gothic architecture?",
     [("Pointed arches — distribute weight more efficiently than rounded arches.",
       "Flying buttresses — external supports that allow taller, thinner walls.",
       "Ribbed vaults — intricate ceiling patterns providing structural support.",
       "Large stained glass windows — flood interiors with colored light.",
       "Gargoyles and grotesques — decorative waterspouts and ornamental figures.",
       "Tall, soaring spires and towers — emphasizing vertical aspiration.",
       "Elaborate stone tracery — decorative stonework in windows and walls."),
      ]),

    # Philosophy
    ("philosophy", "What are the main branches of philosophy?",
     [("Metaphysics — the study of the nature of reality, existence, and being.",
       "Epistemology — the study of knowledge, belief, and justification.",
       "Ethics — the study of morality, values, and how to live well.",
       "Logic — the study of valid reasoning and argumentation.",
       "Aesthetics — the study of beauty, art, and taste.",
       "Political philosophy — the study of governance, justice, and rights.",
       "Philosophy of mind — the study of consciousness and mental phenomena."),
      ]),

    # Astronomy
    ("astronomy", "List the main types of galaxies.",
     [("Spiral galaxies — flat, rotating discs with curved arms (e.g., the Milky Way).",
       "Barred spiral galaxies — spiral galaxies with a bar-shaped center.",
       "Elliptical galaxies — smooth, featureless, ranging from nearly spherical to elongated.",
       "Irregular galaxies — no defined shape, often the result of galactic collisions.",
       "Lenticular galaxies — disc-shaped like spirals but without spiral arms.",
       "Dwarf galaxies — small galaxies that orbit larger ones."),
      ]),

    # Medicine
    ("medicine", "What are the stages of wound healing?",
     [("Hemostasis — blood vessels constrict and platelets form a clot to stop bleeding.",
       "Inflammation — immune cells clear debris and bacteria from the wound.",
       "Proliferation — new tissue, collagen, and blood vessels form.",
       "Remodeling — scar tissue matures and strengthens over weeks to months."),
      ]),
    ("medicine", "List the key risk factors for cardiovascular disease.",
     [("High blood pressure — damages artery walls over time.",
       "High LDL cholesterol — builds up as plaque in arteries.",
       "Smoking — damages blood vessels and reduces oxygen supply.",
       "Diabetes — high blood sugar damages arteries and nerves.",
       "Obesity — increases strain on the heart and contributes to other risk factors.",
       "Physical inactivity — weakens the heart and increases risk factors.",
       "Family history — genetic predisposition to heart disease.",
       "Chronic stress — contributes to high blood pressure and unhealthy behaviors."),
      ]),

    # Economics
    ("economics", "What are the main factors of production?",
     [("Land — natural resources used in production (minerals, water, fertile soil).",
       "Labor — the human effort, both physical and mental, applied to production.",
       "Capital — machinery, tools, buildings, and technology used in production.",
       "Entrepreneurship — the ability to combine other factors innovatively.",
       "Information/technology — increasingly recognized as a key modern factor."),
      ]),
    ("economics", "List the key differences between microeconomics and macroeconomics.",
     [("Microeconomics studies individual consumers, firms, and markets; macroeconomics studies entire economies.",
       "Microeconomics focuses on supply and demand for specific goods; macroeconomics on national output and inflation.",
       "Microeconomics analyzes price determination; macroeconomics analyzes the general price level.",
       "Microeconomics examines firm behavior; macroeconomics examines government fiscal and monetary policy.",
       "Microeconomics is 'bottom-up'; macroeconomics is 'top-down'.",
       "Both use similar tools but at different scales of analysis."),
      ]),

    # Language
    ("language", "What are the main language families of the world?",
     [("Indo-European — includes English, Spanish, French, Hindi, and Russian.",
       "Sino-Tibetan — includes Mandarin Chinese and Tibetan.",
       "Afro-Asiatic — includes Arabic, Hebrew, and Amharic.",
       "Niger-Congo — largest by number of languages, including Swahili and Yoruba.",
       "Austronesian — includes Malay, Tagalog, and Malagasy.",
       "Dravidian — includes Tamil, Telugu, and Kannada.",
       "Turkic — includes Turkish, Uzbek, and Kazakh.",
       "Japonic — Japanese and related languages."),
      ]),

    # Mathematics
    ("mathematics", "What are the key branches of mathematics?",
     [("Arithmetic — the study of numbers and basic operations.",
       "Algebra — manipulation of symbols and solving equations.",
       "Geometry — the study of shapes, sizes, and spatial relationships.",
       "Calculus — the study of rates of change and accumulation.",
       "Statistics and probability — data analysis and likelihood.",
       "Number theory — properties and relationships of integers.",
       "Topology — the study of properties preserved under continuous deformation.",
       "Combinatorics — counting, arrangement, and combination of objects."),
      ]),

    # Parenting
    ("parenting", "What are the most important skills to teach children?",
     [("Emotional regulation — recognizing and managing feelings.",
       "Empathy — understanding and sharing the feelings of others.",
       "Responsibility — taking ownership of their actions and belongings.",
       "Problem-solving — thinking through challenges independently.",
       "Resilience — bouncing back from setbacks and failures.",
       "Communication — expressing thoughts clearly and listening actively.",
       "Critical thinking — questioning assumptions and evaluating evidence.",
       "Basic life skills — cooking, budgeting, and personal hygiene."),
      ]),

    # Fashion
    ("fashion", "What are the wardrobe essentials every person should own?",
     [("A well-fitted pair of dark jeans — versatile for casual and smart-casual occasions.",
       "A classic white button-down shirt — works for almost any setting.",
       "A tailored blazer — elevates any outfit instantly.",
       "A quality pair of leather shoes or clean white sneakers.",
       "A neutral-colored crewneck or V-neck sweater.",
       "A simple black dress or well-cut trousers.",
       "A light trench coat or structured jacket for layering.",
       "Quality basics — plain white and grey t-shirts in good fabrics."),
      ]),

    # Gardening
    ("gardening", "What are the steps to start a vegetable garden?",
     [("Choose a sunny location that gets at least 6 hours of direct sunlight daily.",
       "Test your soil and amend with compost or organic matter.",
       "Decide what to grow based on your climate and preferences.",
       "Plan the layout, considering plant spacing and companion planting.",
       "Start seeds indoors or purchase seedlings from a nursery.",
       "Plant at the right time according to your local frost dates.",
       "Water consistently, targeting the roots rather than the leaves.",
       "Fertilize regularly and watch for pests and diseases."),
      ]),
    ("gardening", "List the easiest vegetables to grow for beginners.",
     [("Lettuce — grows quickly, tolerates partial shade, and can be harvested continuously.",
       "Radishes — ready to harvest in as little as 3-4 weeks.",
       "Zucchini — prolific producer with minimal care needed.",
       "Green beans — easy to grow and high yielding.",
       "Cherry tomatoes — more forgiving than large tomatoes.",
       "Kale — hardy, nutritious, and tolerates cold.",
       "Herbs like basil, parsley, and chives — great for containers."),
      ]),
]


DISH_NAMES = ["lasagna", "chicken curry", "beef stew", "lentil soup", "chocolate cake",
              "paella", "pad thai", "ramen", "biryani", "shakshuka"]
CITY_NAMES = ["Paris", "Rome", "Barcelona", "Tokyo", "New York", "London",
              "Amsterdam", "Prague", "Istanbul", "Lisbon"]
SPORT_NAMES = ["soccer", "tennis", "basketball", "baseball", "volleyball",
               "rugby", "cricket", "golf"]
DESTINATION_NAMES = ["Southeast Asia", "the Mediterranean", "South America",
                     "East Africa", "the Pacific Northwest", "Northern Europe"]


def fill_template(prompt_template):
    t = prompt_template
    if "{dish}" in t:
        t = t.replace("{dish}", random.choice(DISH_NAMES))
    if "{city}" in t:
        t = t.replace("{city}", random.choice(CITY_NAMES))
    if "{sport}" in t:
        t = t.replace("{sport}", random.choice(SPORT_NAMES))
    if "{destination}" in t:
        t = t.replace("{destination}", random.choice(DESTINATION_NAMES))
    return t


def make_pair(prompt_template, items):
    prompt = fill_template(prompt_template)

    # Build rejected (dash markers)
    rejected_lines = [f"- {item}" for item in items]
    rejected = "\n".join(rejected_lines)

    # Build chosen ((*) markers)
    chosen_lines = [f"(*) {item}" for item in items]
    chosen = "\n".join(chosen_lines)

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main():
    output_path = "/weka/nora-default/varshak/critic_lfqa/small_synthetic_exp/dpo_dataset.jsonl"
    target = 2000
    records = []
    category_counts = {}

    while len(records) < target:
        cat, prompt_tmpl, items_list = random.choice(TEMPLATES)
        items = random.choice(items_list)
        pair = make_pair(prompt_tmpl, items)

        records.append(pair)
        category_counts[cat] = category_counts.get(cat, 0) + 1

        if len(records) % 200 == 0:
            print(f"Generated {len(records)}/{target} records...")

    # Write JSONL
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Stats
    avg_chosen_len = sum(len(r["chosen"]) for r in records) / len(records)
    avg_rejected_len = sum(len(r["rejected"]) for r in records) / len(records)

    print(f"\nDone! Saved {len(records)} records to {output_path}")
    print(f"Average chosen response length: {avg_chosen_len:.0f} chars")
    print(f"Average rejected response length: {avg_rejected_len:.0f} chars")
    print("\nTopic distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
