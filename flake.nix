{
  description = "PyTorch dev environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        pkgs.python310Full
        pkgs.python310Packages.virtualenv
        pkgs.git
        pkgs.cmake
        pkgs.gcc
      ];

      shellHook = ''
        if [ ! -d ".venv" ]; then
          python -m venv .venv
        fi
        source .venv/bin/activate

        pip install --upgrade pip
        pip install torch --index-url https://download.pytorch.org/whl/cpu
      '';
    };
  };
}

