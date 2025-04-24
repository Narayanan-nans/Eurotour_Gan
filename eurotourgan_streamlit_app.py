import streamlit as st
import numpy as np
import torch
import random
import json
from eurotourgan_gan import Generator

# Load JSON data
with open("gan_flat_dataset.json", "r") as f:
    json_data = json.load(f)

@st.cache_resource
def load_generator():
    model = Generator(input_dim=20, output_dim=50) 
    try:
        state_dict = torch.load("generator.pt", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        st.success("✅ Trained model loaded successfully.")
    except FileNotFoundError:
        st.warning("⚠️ Trained model not found. Using untrained model.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    model.eval()
    return model


def encode_user_input(destination, season, days, budget, selected_interests):
    destinations = ['Paris', 'London', 'Rome', 'Barcelona', 'Amsterdam']
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    interests = ['Art', 'Food', 'Shopping', 'Nightlife', 'Beaches', 'Museums', 'Photography', 'Adventure Sports', 'Cultural Heritage']

    dest_vec = [1 if destination == d else 0 for d in destinations]
    norm_days = [days / 30.0]
    norm_budget = [budget / 500000.0]
    season_vec = [1 if season == s else 0 for s in seasons]
    interest_vec = [1 if i in selected_interests else 0 for i in interests]

    input_vector = dest_vec + norm_days + norm_budget + season_vec + interest_vec
    if len(input_vector) != 20:
        raise ValueError(f"❌ Input vector length is {len(input_vector)} instead of 20.")
    return input_vector


def get_real_activity(destination, interest):
    matches = [entry for entry in json_data if entry["country"].lower() == destination.lower() and interest.lower() in entry["interest"].lower()]
    if matches:
        return random.choice(matches)
    else:
        return {"description": f"Explore {interest} activities in {destination}.", "price": random.randint(7500, 20000)}

def decode_itinerary_vector(user_inputs, vector, days):
    times_of_day = ["Morning", "Afternoon", "Evening"]
    intensity_labels = ["Relaxed", "Moderate", "Intense"]
    interests = ['Art', 'Food', 'Shopping', 'Nightlife', 'Beaches', 'Museums', 'Photography', 'Adventure Sports', 'Cultural Heritage']

    itinerary = []
    chunk_size = 5
    max_days_from_vector = len(vector) // chunk_size
    days = min(days, max_days_from_vector)

    selected_interests = user_inputs[3]
    for i in range(days):
        base = i * chunk_size
        gan_intensity = vector[base + 3]
        gan_time = vector[base + 4]
        time_of_day = times_of_day[int(gan_time * 3) % 3]
        interest = selected_interests[i % len(selected_interests)]
        real_activity = get_real_activity(user_inputs[0], interest)

        activity = f"Day {i+1}: In {user_inputs[0]} during the {user_inputs[1]}, enjoy **{interest}** in the **{time_of_day}** with a **{intensity_labels[int(gan_intensity * 3) % 3]}** pace. \n> _{real_activity['description']}_\nEstimated cost: ₹{real_activity['price']}"
        itinerary.append(activity)

    return itinerary


st.set_page_config(page_title="EuroTourGAN", page_icon="🌍")
st.title("🌍 EuroTourGAN: AI Travel Itinerary Planner")
st.write("Plan your perfect Europe trip based on interests, season, and budget.")

destination = st.selectbox("Choose your destination", ['Paris', 'London', 'Rome', 'Barcelona', 'Amsterdam'])
season = st.selectbox("Preferred season", ['Spring', 'Summer', 'Autumn', 'Winter'])
days = st.slider("Trip duration (days)", 1, 15, 5)
budget = st.number_input("Your budget (in INR)", min_value=10000, max_value=500000, value=100000, step=1000)
selected_interests = st.multiselect("Select your interests", ['Art', 'Food', 'Shopping', 'Nightlife',  'Museums', 'Photography', 'Adventure Sports', 'Cultural Heritage'])

if st.button("🎒 Generate Itinerary"):
    if not selected_interests:
        st.error("Please select at least one interest.")
    else:
        try:
            user_vector = encode_user_input(destination, season, days, budget, selected_interests)
            input_tensor = torch.tensor([user_vector], dtype=torch.float32)

            generator = load_generator()
            itinerary_vector = generator(input_tensor).detach().numpy().flatten()

            user_inputs = (destination, season, budget, selected_interests)
            itinerary = decode_itinerary_vector(user_inputs, itinerary_vector, days)

            st.subheader("📅 Your Personalized Itinerary")
            st.markdown(f"**Destination:** {destination}")
            st.markdown(f"**Season:** {season}")
            st.markdown(f"**Duration:** {days} days")
            st.markdown(f"**Budget:** ₹{budget}")
            st.markdown(f"**Your Interests:** {', '.join(selected_interests)}")

            for day_plan in itinerary:
                st.write(day_plan)

        except Exception as e:
            st.error(f"❌ Failed to generate itinerary: {e}")
