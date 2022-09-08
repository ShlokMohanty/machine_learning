from flask import Flask, request 
app=Flask("hello request")
@app.route("/home/")
def hello() 
{
return str(request.args)
}
app.run()
