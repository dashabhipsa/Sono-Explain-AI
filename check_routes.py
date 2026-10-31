from app.backend.main import app

print("\nRegistered API routes:\n")

for route in app.routes:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)

    if path:
        print(path, methods)
    else:
        print(type(route).__name__)