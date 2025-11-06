{
  pkgs ? import <nixpkgs> { },
  ...
}:
(pkgs.buildFHSEnv {
  name = "python";
  targetPkgs =
    pkgs: with pkgs; [
      glibc
      uv
      nodejs
    ];
  extraOutputsToInstall = [ "dev" ];
}).env
