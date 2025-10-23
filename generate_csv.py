import pandas as pd
import random

areas = [
    # --- Delhi (10+) ---
    ("Connaught Place", "Delhi", 28.6315, 77.2167),
    ("India Gate", "Delhi", 28.6129, 77.2295),
    ("Saket", "Delhi", 28.5245, 77.2060),
    ("Dwarka Sector 12", "Delhi", 28.5876, 77.0460),
    ("Lajpat Nagar", "Delhi", 28.5686, 77.2439),
    ("Karol Bagh", "Delhi", 28.6550, 77.1900),
    ("Hauz Khas", "Delhi", 28.5494, 77.2001),
    ("Rajouri Garden", "Delhi", 28.6422, 77.1174),
    ("Vasant Kunj", "Delhi", 28.5205, 77.1554),
    ("Janakpuri", "Delhi", 28.6219, 77.0922),

    # --- Noida (8+) ---
    ("Noida Sector 18", "Noida", 28.5707, 77.3260),
    ("Noida Sector 62", "Noida", 28.6304, 77.3733),
    ("Noida Sector 15", "Noida", 28.5839, 77.3147),
    ("Botanical Garden", "Noida", 28.5640, 77.3348),
    ("Film City", "Noida", 28.5681, 77.3312),
    ("Noida Sector 128", "Noida", 28.5200, 77.3640),
    ("Noida Sector 104", "Noida", 28.5540, 77.3645),
    ("Noida Sector 76", "Noida", 28.5940, 77.3800),

    # --- Ghaziabad (7+) ---
    ("Raj Nagar", "Ghaziabad", 28.6670, 77.4440),
    ("Kaushambi", "Ghaziabad", 28.6430, 77.3283),
    ("Indirapuram", "Ghaziabad", 28.6437, 77.3727),
    ("Vaishali", "Ghaziabad", 28.6438, 77.3415),
    ("Crossing Republik", "Ghaziabad", 28.6375, 77.4000),
    ("Modinagar", "Ghaziabad", 28.8330, 77.5670),
    ("Vasundhara", "Ghaziabad", 28.6591, 77.3641),

    # --- Greater Noida (6+) ---
    ("Pari Chowk", "Greater Noida", 28.4744, 77.5030),
    ("Alpha 1", "Greater Noida", 28.4702, 77.5095),
    ("Gamma 2", "Greater Noida", 28.4775, 77.4988),
    ("Knowledge Park", "Greater Noida", 28.4682, 77.4911),
    ("Delta 1", "Greater Noida", 28.4790, 77.5205),
    ("Zeta 2", "Greater Noida", 28.4480, 77.5130),

    # --- Gurgaon (8+) ---
    ("Cyber Hub", "Gurgaon", 28.4945, 77.0880),
    ("MG Road", "Gurgaon", 28.4711, 77.0786),
    ("DLF Phase 3", "Gurgaon", 28.4839, 77.1015),
    ("Sohna Road", "Gurgaon", 28.4029, 77.0427),
    ("Golf Course Road", "Gurgaon", 28.4522, 77.0970),
    ("Sector 29", "Gurgaon", 28.4672, 77.0732),
    ("Udyog Vihar", "Gurgaon", 28.5005, 77.0878),
    ("Manesar", "Gurgaon", 28.3577, 76.9397),
]

data = []
for name, city, lat, lon in areas:
    data.append({
        "city": city,
        "area": name,
        "latitude": lat,
        "longitude": lon,
        "reports": random.randint(1, 30),
        "lighting": round(random.uniform(2, 5), 1),
        "cctv": round(random.uniform(1, 5), 1),
        "crowd_density": round(random.uniform(1, 5), 1),
        "user_rating": round(random.uniform(2.5, 5), 1)
    })

df = pd.DataFrame(data)
df.to_csv("safety_data.csv", index=False)
print(f"✅ safety_data.csv created successfully with {len(df)} locations!")
