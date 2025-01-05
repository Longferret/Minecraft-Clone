# Main crate - [`main.rs`](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Minecraft-v0.3/main_app/src/main.rs)
This document contains all information on the main crate.


The main crate only contains the main loop, which does input detection and sequentially updates both the rendering system and the game core. 

This loop ensures that the game progresses by continuously updating the game state and
rendering the changes to the screen.