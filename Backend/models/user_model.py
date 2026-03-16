from database import users_collection

def get_user_by_email(email):
    return users_collection.find_one({"email": email})


def create_user(name, email, password, role):

    user = {
        "name": name,
        "email": email,
        "password": password,
        "role": role
    }

    result = users_collection.insert_one(user)
    return str(result.inserted_id)