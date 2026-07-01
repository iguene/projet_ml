// gestion de la base SQLite
// sert a garder une trace des entrainements (regression lineaire et MLP)
// avec leur precision, pour pouvoir comparer les sessions plus tard

use rusqlite::{params, Connection};
use std::time::{SystemTime, UNIX_EPOCH};

// ouvre (ou cree) le fichier de base et prepare la table
pub fn ouvrir_base(chemin: &str) -> Connection {
    let conn = Connection::open(chemin).unwrap();
    creer_table(&conn);
    conn
}

fn creer_table(conn: &Connection) {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS entrainements (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            type_modele TEXT NOT NULL,
            poids       TEXT NOT NULL,
            precision   REAL NOT NULL,
            date_creation INTEGER NOT NULL
        )",
        [],
    ).unwrap();
}

fn maintenant() -> i64 {
    let duree = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    duree.as_secs() as i64
}

// sauvegarde une session d'entrainement (modele deja serialise en json + sa precision)
pub fn sauvegarder_session(conn: &Connection, type_modele: &str, poids_json: &str, precision: f64) -> i64 {
    conn.execute(
        "INSERT INTO entrainements (type_modele, poids, precision, date_creation) VALUES (?1, ?2, ?3, ?4)",
        params![type_modele, poids_json, precision, maintenant()],
    ).unwrap();

    conn.last_insert_rowid()
}

// liste l'historique des entrainements (id, type, precision)
pub fn lister_sessions(conn: &Connection) -> Vec<(i64, String, f64)> {
    let mut requete = conn.prepare(
        "SELECT id, type_modele, precision FROM entrainements ORDER BY id DESC"
    ).unwrap();

    let lignes = requete.query_map([], |ligne| {
        let id: i64 = ligne.get(0)?;
        let type_modele: String = ligne.get(1)?;
        let precision: f64 = ligne.get(2)?;
        Ok((id, type_modele, precision))
    }).unwrap();

    let mut sessions = Vec::new();
    for ligne in lignes {
        sessions.push(ligne.unwrap());
    }
    sessions
}
