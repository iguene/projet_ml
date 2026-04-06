mod models;
mod data;

use models::linear::LinearModel;
use models::mlp::MLP;
use data::{load_dataset, print_dataset};
use std::io::{self, Write};

fn lire_f64(prompt: &str) -> f64 {
    loop {
        print!("  {}", prompt);
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        match input.trim().parse::<f64>() {
            Ok(v) => return v,
            Err(_) => println!("  Valeur invalide, réessayez."),
        }
    }
}

fn menu() -> String {
    println!("\n╔══════════════════════════════════════╗");
    println!("║     APPLICATION ML — StarCraft II    ║");
    println!("╠══════════════════════════════════════╣");
    println!("║  1. Afficher le dataset              ║");
    println!("║  2. Entraîner — Régression Linéaire  ║");
    println!("║  3. Entraîner — MLP                  ║");
    println!("║  4. Inférence manuelle               ║");
    println!("║  5. Quitter                          ║");
    println!("╚══════════════════════════════════════╝");
    print!("Votre choix : ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn inference_manuelle(linear: &Option<LinearModel>, mlp: &Option<MLP>) {
    println!("\n--- Inférence manuelle ---");
    let apm        = lire_f64("APM (ex: 95) : ");
    let minerals   = lire_f64("Minerals (ex: 3000) : ");
    let gas        = lire_f64("Gas (ex: 1200) : ");
    let units_lost = lire_f64("Units Lost (ex: 25) : ");

    let norm_vec = vec![apm/200.0, minerals/5000.0, gas/2500.0, units_lost/100.0];
    let norm_arr = [apm/200.0, minerals/5000.0, gas/2500.0, units_lost/100.0];

    println!("\n  Résultats :");
    if let Some(lin) = linear {
        let pred = lin.predict(&norm_vec);
        println!("  [Linéaire] Score: {:.3} → {}", pred,
                 if pred >= 0.5 { "VICTOIRE ✓" } else { "DÉFAITE ✗" });
    } else {
        println!("  [Linéaire] Modèle non entraîné — choisir option 2 d'abord");
    }
    if let Some(m) = mlp {
        let (pred, _) = m.forward(&norm_arr);
        println!("  [MLP]      Score: {:.3} → {}", pred,
                 if pred >= 0.5 { "VICTOIRE ✓" } else { "DÉFAITE ✗" });
    } else {
        println!("  [MLP] Modèle non entraîné — choisir option 3 d'abord");
    }
}

fn main() {
    let dataset = load_dataset();
    let mut linear_model: Option<LinearModel> = None;
    let mut mlp_model: Option<MLP> = None;

    loop {
        let choix = menu();
        match choix.as_str() {
            "1" => {
                println!("\n=== DATASET SC2 ({} parties) ===\n", dataset.len());
                print_dataset(&dataset);
            }
            "2" => {
                println!("\n=== ENTRAÎNEMENT — RÉGRESSION LINÉAIRE ===\n");
                let data_lin: Vec<(Vec<f64>, f64)> = dataset.iter()
                    .map(|g| (g.to_normalized_vec(), g.win))
                    .collect();
                let mut model = LinearModel::new(4);
                model.train(&data_lin, 0.1, 1000);
                println!("\n  Précision : {:.0}%", model.accuracy(&data_lin));
                linear_model = Some(model);
            }
            "3" => {
                println!("\n=== ENTRAÎNEMENT — MLP ===\n");
                let data_mlp: Vec<([f64; 4], f64)> = dataset.iter()
                    .map(|g| (g.to_normalized_array(), g.win))
                    .collect();
                let mut model = MLP::new(123);
                model.train(&data_mlp, 0.5, 2000);
                println!("\n  Précision : {:.0}%", model.accuracy(&data_mlp));
                mlp_model = Some(model);
            }
            "4" => {
                inference_manuelle(&linear_model, &mlp_model);
            }
            "5" => {
                println!("\nAu revoir !\n");
                break;
            }
            _ => println!("  Choix invalide."),
        }
    }
}