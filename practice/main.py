from fastapi import FastAPI

app = FastAPI()

candidates = [
    {
        "id": 1,
        "name": "Vicky",
        "age": 22,
        "skill": "Python",
        "experience": 2,
        "city": "Mumbai"
    },
    {
        "id": 2,
        "name": "Rahul",
        "age": 25,
        "skill": "Java",
        "experience": 3,
        "city": "Pune"
    },
    {
        "id": 3,
        "name": "Priya",
        "age": 23,
        "skill": "Python",
        "experience": 1,
        "city": "Delhi"
    },
    {
        "id": 4,
        "name": "Ankit",
        "age": 28,
        "skill": "C++",
        "experience": 5,
        "city": "Mumbai"
    }
]

@app.get("/candidate/{candidate_id}")
def get_candidate(candidate_id: int):
        
    for candidate in candidates:
        if candidate["id"] == candidate_id:
            return candidate
        
    return(f"Error: No candidate found with id {candidate_id}")