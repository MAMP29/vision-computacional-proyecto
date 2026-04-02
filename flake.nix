{
  description = "Entorno Python con UV y soporte CUDA";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
      
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
        glib
        libglvnd
        cudaPackages.cuda_cudart
        cudaPackages.libcublas
        cudaPackages.cuda_nvrtc
        cudaPackages.libcufft
        cudaPackages.libcurand
        cudaPackages.cudnn
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv
        ];

        env = {
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath runtimeLibs}:/run/opengl-driver/lib:/run/lib64";
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = "${pkgs.python312}/bin/python3";
        };

        shellHook = ''
          if [ ! -d ".venv" ]; then
            uv venv
          fi
          source .venv/bin/activate
          
          echo "🔥 Entorno PyTorch + UV listo"
          python -c "import torch; print(f'¿CUDA disponible?: {torch.cuda.is_available()}'); print(f'Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Ninguno\"}')"
        '';
      };
    };
}
