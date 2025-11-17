#include "rdr/integrator.h"

#include <cmath>
#include <iterator>
#include <math.h>
#include <omp.h>
#include <ostream>
#include <iostream>

#include "linalg.h"
#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        //
        // Useful Functions:
        //
        // @see Sampler::getPixelSample for getting the current pixel sample
        // as Vec2f.
        //
        // @see Camera::generateDifferentialRay for generating rays given
        // pixel sample positions as 2 floats.

        // You should assign the following two variables
        // const Vec2f &pixel_sample = ...
        // auto ray = ...

        // After you assign pixel_sample and ray, you can uncomment the
        // following lines to accumulate the radiance to the film.
        const Vec2f &pixel_sample  = sampler.getPixelSample();
        auto ray = camera->generateDifferentialRay(pixel_sample.x,pixel_sample.y);

        // Accumulate radiance
        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction      = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    if (is_perfect_refraction) {
      // We should follow the specular direction
      // TODO(HW3): call the interaction.bsdf->sample to get the new direction
      // and update the ray accordingly.
      //
      // Useful Functions:
      // @see BSDF::sample
      // @see SurfaceInteraction::spawnRay
      //
      // You should update ray = ... with the spawned ray
      interaction.bsdf->sample(interaction, sampler, nullptr);
      ray = interaction.spawnRay(interaction.wi);
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction);
  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  // std::cout << point_light_position << std::endl;
  Vec3f color(0, 0, 0);
  for (int i = 0; i < point_light_positions.size(); i++) 
  {
    Vec3f point_light_position = point_light_positions[i];
    Vec3f point_light_flux     = point_light_fluxs[i];
    Float dist_to_light = Norm(point_light_position - interaction.p);
    Vec3f light_dir     = Normalize(point_light_position - interaction.p);
    auto test_ray       = DifferentialRay(interaction.p, light_dir);
    interaction.wi = light_dir;

    // TODO(HW3): Test for occlusion
    //
    // You should test if there is any intersection between interaction.p and
    // point_light_position using scene->intersect. If so, return an occluded
    // color. (or Vec3f color(0, 0, 0) to be specific)
    //
    // You may find the following variables useful:
    //
    // @see bool Scene::intersect(const Ray &ray, SurfaceInteraction &interaction)
    //    This function tests whether the ray intersects with any geometry in the
    //    scene. And if so, it returns true and fills the interaction with the
    //    intersection information.
    //
    //    You can use iteraction.p to get the intersection position.
    //
    SurfaceInteraction newSurface;
    bool intersected = scene->intersect(test_ray, newSurface);
    Vec3f interactPoint = newSurface.p;
    if (linalg::distance(interactPoint, test_ray.origin) < linalg::distance(point_light_position, test_ray.origin) - 1e-4 && intersected)
    {
      continue;
    }
    // SurfaceInteraction shadow_interaction;
    // test_ray.t_max = dist_to_light * (1.0f - 1e-4f); 
    // if (scene->intersect(test_ray, shadow_interaction)) {
    //     return color;
    // }
    // Not occluded, compute the contribution using perfect diffuse diffuse model
    // Perform a quick and dirty check to determine whether the BSDF is ideal
    // diffuse by RTTI
    const BSDF *bsdf      = interaction.bsdf;
    bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

    if (bsdf != nullptr && is_ideal_diffuse) {
      // TODO(HW3): Compute the contribution
      //
      // You can use bsdf->evaluate(interaction) * cos_theta to approximate the
      // albedo. In this homework, we do not need to consider a
      // radiometry-accurate model, so a simple phong-shading-like model is can be
      // used to determine the value of color.

      // The angle between light direction and surface normal
      Float cos_theta = std::max(Dot(light_dir, interaction.normal), 0.0f);

      // if (dist_to_light < 1) {
      //   dist_to_light = 1;
      // }
      Vec3f albedo = bsdf->evaluate(interaction) * cos_theta;
      Vec3f radiance = point_light_flux / (4 * PI * dist_to_light * dist_to_light);
      // Vec3f radiance = point_light_flux *(dist_to_light * dist_to_light)/(4*PI);
      // std::cout<< radiance[0]<< " ," << radiance[1]<< " ," << radiance[2]<< std::endl;
      color +=  radiance * albedo;
      // color = bsdf->evaluate(interaction) * cos_theta;
    }
  }
  return color;
}


void AreaLightTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        //
        // Useful Functions:
        //
        // @see Sampler::getPixelSample for getting the current pixel sample
        // as Vec2f.
        //
        // @see Camera::generateDifferentialRay for generating rays given
        // pixel sample positions as 2 floats.

        // You should assign the following two variables
        // const Vec2f &pixel_sample = ...
        // auto ray = ...

        // After you assign pixel_sample and ray, you can uncomment the
        // following lines to accumulate the radiance to the film.
        const Vec2f &pixel_sample  = sampler.getPixelSample();
        auto ray = camera->generateDifferentialRay(pixel_sample.x,pixel_sample.y);

        // Accumulate radiance
        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f AreaLightTestIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction      = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    if (is_perfect_refraction) {
      // We should follow the specular direction
      // TODO(HW3): call the interaction.bsdf->sample to get the new direction
      // and update the ray accordingly.
      //
      // Useful Functions:
      // @see BSDF::sample
      // @see SurfaceInteraction::spawnRay
      //
      // You should update ray = ... with the spawned ray
      interaction.bsdf->sample(interaction, sampler, nullptr);
      ray = interaction.spawnRay(interaction.wi);
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction);
  return color;
}

Vec3f AreaLightTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  // std::cout << "percessing" << std::endl;
  vector<Vec3f> colors;
  Sampler sampler;
  Vec3f final_color(0,0,0);
  for (int j = 0; j < scene->getLights().size(); j++){
    for (int i = 0; i < spp; i++) {
      Vec3f color(0, 0, 0);
      SurfaceInteraction light_interaction = scene->getLights()[j]->sample(interaction, sampler);
      Vec3f light_dir = Normalize(light_interaction.p - interaction.p);
      Float dist_to_light = Norm(light_interaction.p - interaction.p);
      auto test_ray       = DifferentialRay(interaction.p, light_dir);

      if (dynamic_cast<AreaLight *>(scene->getLights()[j].get()) != nullptr)
      {
        // std::cout << "area light" << std::endl;
        SurfaceInteraction newSurface;
        bool intersected = scene->intersect(test_ray, newSurface);
        Vec3f interactPoint = newSurface.p;
        if (linalg::distance(interactPoint, test_ray.origin) < linalg::distance(light_interaction.p, test_ray.origin) - 1e-4 && intersected)
        {
          colors.push_back(color);
          continue;
        }
      }
      else {
        // std::cout << "infinite light" << std::endl;
        SurfaceInteraction newSurface;
        bool intersected = scene->intersect(test_ray, newSurface);
        if (intersected)
        {
          colors.push_back(color);
          continue;
        }
      }

      const BSDF *bsdf      = interaction.bsdf;
      bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

      if (bsdf != nullptr && is_ideal_diffuse) {

        Float cos_theta = std::max(Dot(light_dir, interaction.normal), 0.0f);
        
        Vec3f albedo = bsdf->evaluate(interaction) * cos_theta;
        if (dynamic_cast<AreaLight *>(scene->getLights()[j].get()) != nullptr)
        {
          Float light_pdf = light_interaction.pdf;
          Vec3f radiance = scene->getLights()[j]->Le(light_interaction, -light_dir) * Dot(-light_dir, light_interaction.normal) / (light_pdf * dist_to_light * dist_to_light);
          color =  radiance * albedo;
          colors.push_back(color);
        }
        else if (dynamic_cast<InfiniteAreaLight *>(scene->getLights()[j].get()) != nullptr)
        {
          // std::cout << "infinite light" << std::endl;
          Float light_pdf = light_interaction.pdf;
          Vec3f radiance = scene->getLights()[j]->Le(light_interaction, -light_dir) / light_pdf;
          color =  radiance * albedo;
          colors.push_back(color);
        }
        // Vec3f radiance = scene->getLights()[j]->Le(light_interaction, -light_dir) * Dot(-light_dir, light_interaction.normal) / (light_interaction.pdf * dist_to_light * dist_to_light);
        // color =  radiance * albedo;
        // colors.push_back(color);
      }
    }
    for (auto color : colors) {
      final_color += color;
    }
  }
  return final_color / Float(spp * scene->getLights().size());
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
  return Vec3f(0.0);
}

Vec3f PathIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
  return Vec3f(0.0);
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li(  // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
  return Vec3f(0.0);
}

RDR_NAMESPACE_END
