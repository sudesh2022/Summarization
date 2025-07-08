
Application components
1. Setup olama on your local machine 
2. Setup postgres on your local machine 
3. The purpose of the database is to store existing summary so that retreival is faster
4. 

Running the app :
nohup uvicorn fastOllama:app --reload --host 0.0.0.0 --port 8000 > server.log

If we have to shut down the app :

lsof -i :8000

Find the PID (process ID) in the output, then run:kill <PID>