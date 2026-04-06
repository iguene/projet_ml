pub struct GameData {
    pub apm: f64,
    pub minerals: f64,
    pub gas: f64,
    pub units_lost: f64,
    pub win: f64,
}

impl GameData {
    pub fn to_normalized_vec(&self) -> Vec<f64> {
        vec![
            self.apm        / 200.0,
            self.minerals   / 5000.0,
            self.gas        / 2500.0,
            self.units_lost / 100.0,
        ]
    }

    pub fn to_normalized_array(&self) -> [f64; 4] {
        [
            self.apm        / 200.0,
            self.minerals   / 5000.0,
            self.gas        / 2500.0,
            self.units_lost / 100.0,
        ]
    }
}

pub fn load_dataset() -> Vec<GameData> {
    vec![
        GameData { apm: 120.0, minerals: 4200.0, gas: 1800.0, units_lost: 15.0, win: 1.0 },
        GameData { apm: 60.0,  minerals: 1200.0, gas: 400.0,  units_lost: 42.0, win: 0.0 },
        GameData { apm: 95.0,  minerals: 3100.0, gas: 900.0,  units_lost: 20.0, win: 1.0 },
        GameData { apm: 45.0,  minerals: 800.0,  gas: 200.0,  units_lost: 55.0, win: 0.0 },
        GameData { apm: 110.0, minerals: 3800.0, gas: 1500.0, units_lost: 18.0, win: 1.0 },
        GameData { apm: 70.0,  minerals: 1500.0, gas: 500.0,  units_lost: 38.0, win: 0.0 },
        GameData { apm: 130.0, minerals: 4500.0, gas: 2000.0, units_lost: 10.0, win: 1.0 },
        GameData { apm: 50.0,  minerals: 1000.0, gas: 300.0,  units_lost: 48.0, win: 0.0 },
    ]
}

pub fn print_dataset(data: &Vec<GameData>) {
    println!("  {:<6} {:<10} {:<6} {:<12} {:<8}", "APM", "Minerals", "Gas", "Units Lost", "Victoire");
    println!("  {}", "-".repeat(46));
    for g in data {
        println!("  {:<6} {:<10} {:<6} {:<12} {}",
            g.apm, g.minerals, g.gas, g.units_lost,
            if g.win == 1.0 { "✓ Victoire" } else { "✗ Défaite" });
    }
}