import MySQLdb

def connect():
    db = MySQLdb.connect(user="tobin",db="cc",passwd="creatingkillerwebsites")
    db.autocommit(True)
    return db.cursor()
