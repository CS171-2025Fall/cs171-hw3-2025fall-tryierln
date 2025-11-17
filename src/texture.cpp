#include "rdr/texture.h"

#include <tinyexr.h>

#include "rdr/interaction.h"
#include "rdr/mipmap.h"
#include <iostream>
RDR_NAMESPACE_BEGIN

Vec2f UVMapping2D::Map(
    const SurfaceInteraction &interaction, Vec2f &dstdx, Vec2f &dstdy) const {
  dstdx = scale * Vec2f(interaction.dudx, interaction.dvdx);
  dstdy = scale * Vec2f(interaction.dudy, interaction.dvdy);
  return scale * interaction.uv + delta;
}

ImageTexture::ImageTexture(const Properties &props) : Texture(props) {
  auto path = props.getProperty<std::string>("path");
  path      = FileResolver::resolveToAbs(path);

  texmap = RDR_CREATE_CLASS(TexCoordinateGenerator,
      props.getProperty<Properties>("tex_coordinate_generator"));

  float *out;
  const char *err = nullptr;
  // std::cout << "Loading texture image from " << path << " ..." << std::endl;
  Info_("Start loading texture image [ {} ]...", path);
  int ret = LoadEXR(&out, &width, &height, path.c_str(), &err);
  // std::cout << "Finished loading texture image from " << path << std::endl;
  Info_("Finished loading texture image [ {} ]", path);
  if (ret != TINYEXR_SUCCESS) {
    // std::cout << "Failed to load texture " << err << std::endl;
    Exception_("Failed to load texture {}", err);
    FreeEXRErrorMessage(err);

  } else {
    // std::cout << "Building MIPMap for texture image from " << path << std::endl;
    data = vector<Float>(out, out + 4 * width * height);
    free(out);
    // std::cout << "Finished building MIPMap for texture image from " << path
              // << std::endl;
    Info_("Start building MIPMap of [ {} ]...", path);
    mipmap = make_ref<MIPMap>(Vec2u(width, height), data);
    Info_("Finished building MIPMap");
  }
}

Vec3f ImageTexture::evaluate(const SurfaceInteraction &interaction) const {
  Vec2f dstdx, dstdy;
  const auto &st = texmap->Map(interaction, dstdx, dstdy);
  return mipmap->LookUp(st, dstdx, dstdy);
}

RDR_NAMESPACE_END
