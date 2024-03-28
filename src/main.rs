use std::{env, net::SocketAddr};

fn main() -> mkdb::Result<()> {
    let file = env::args().nth(1).expect("database file not provided");

    let port = env::args()
        .nth(2)
        .map(|port| port.parse::<u16>().expect("incorrect port number"))
        .unwrap_or(8000);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    mkdb::tcp::server::start(addr, file)
}
