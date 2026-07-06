/*

Copyright (c) 2005-2023, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/**
 * @file
 *
 * This file gives an example of how you can create your own executable
 * in a user project.
 */

#include <iostream>
#include <string>

#include "ExecutableSupport.hpp"
#include "Exception.hpp"
#include "PetscTools.hpp"
#include "PetscException.hpp"

#include "flamegpu/flamegpu.h"

#include "GPUModifier.cuh"
#include "NodesOnlyMesh.hpp"
#include "UniformCellCycleModel.hpp"
#include "OffLatticeSimulation.hpp"
#include "GeneralisedLinearSpringForce.hpp"
#include "CellsGenerator.hpp"
#include "TransitCellProliferativeType.hpp"
#include "SmartPointers.hpp"
#include "SimulationTime.hpp"

#include "Hello_gpu-benchmark-2026.hpp"

enum class SimulatorType {
    CPU,
    GPU
};

typedef struct ResultsRow {
    std::string type;
    std::string dimensions;
    double box_size;
    double mechanics_time;
    double run_time;
} ResultsRow;

using ResultsSet = std::vector<ResultsRow>;

constexpr double END_TIME = 1.0;

void WriteResultsToFile(std::vector<ResultsRow> results, std::string fileName) {
    std::ofstream results_file(fileName);
    for (auto& row : results) {
        results_file << row.type << ", " << row.dimensions << ", " << row.box_size << ", " << row.mechanics_time << ", " << row.run_time << "\n";
    }
    std::cout << "Results written to " << fileName << "\n";
}

template<unsigned DIM>
std::vector<std::array<float, DIM>> RunSim(std::vector<Node<DIM>*> nodes, SimulatorType type) {

    SimulationTime::Destroy();
    SimulationTime::Instance()->SetStartTime(0.0);

    NodesOnlyMesh<DIM> mesh;
    mesh.ConstructNodesWithoutMesh(nodes, 1.5);

    std::vector<CellPtr> cells;
    MAKE_PTR(TransitCellProliferativeType, p_transit_type);
    CellsGenerator<UniformCellCycleModel, DIM> cells_generator;
    cells_generator.GenerateBasicRandom(cells, mesh.GetNumNodes(), p_transit_type);

    NodeBasedCellPopulation<DIM> node_based_cell_population(mesh, cells);

    // Set up cell-based simulation
    OffLatticeSimulation<DIM> simulator(node_based_cell_population);
    if (type == SimulatorType::CPU) {
        simulator.SetOutputDirectory("CPUNodeBased");
    } else if (type == SimulatorType::GPU) {
        simulator.SetOutputDirectory("GPUNodeBased");
    }
    simulator.SetEndTime(END_TIME); // 1 time step

    MAKE_PTR(GPUModifier<DIM>, gpuModifier);
    MAKE_PTR(GeneralisedLinearSpringForce<DIM>, springForce);

    if (type == SimulatorType::CPU) {
        springForce->SetCutOffLength(1.5);
        simulator.AddForce(springForce);
    } else {
        simulator.AddSimulationModifier(gpuModifier);
    }

    // Run simulation
    simulator.Solve();

    // Fetch results
    std::vector<std::array<float, DIM>> locations;
    for (unsigned int i = 0; i < nodes.size(); i++) {
        const auto& loc = mesh.GetNode(0)->rGetLocation();
        if constexpr (DIM == 2) {
            locations.push_back({static_cast<float>(loc[0]), static_cast<float>(loc[1])});
        }
        if constexpr (DIM == 3) {
            locations.push_back({static_cast<float>(loc[0]), static_cast<float>(loc[1]), static_cast<float>(loc[2])});
        }
    }


    // Avoid memory leak
    for (unsigned i=0; i<nodes.size(); i++)
    {
        delete nodes[i];
    }

    return locations;
}

std::vector<std::array<float, 2>> TwoParticlesOutOfRange2DCPU() {
    std::vector<Node<2>*> nodes;
    nodes.push_back(new Node<2>(0, false, -1.0, -0.0));
    nodes.push_back(new Node<2>(1, false, 1.0, 0.0));

    return RunSim<2>(nodes, SimulatorType::CPU);
}


std::vector<std::array<float, 2>> TwoParticlesOutOfRange2DGPU() {

    std::vector<Node<2>*> nodes;
    nodes.push_back(new Node<2>(0, false, -1.0, -0.0));
    nodes.push_back(new Node<2>(1, false, 1.0, 0.0));

    return RunSim<2>(nodes, SimulatorType::GPU);
}

std::vector<std::array<float, 3>> TwoParticlesOutOfRange3DCPU() {
    std::vector<Node<3>*> nodes;
    nodes.push_back(new Node<3>(0, false, -1.0, -0.0, 0.0));
    nodes.push_back(new Node<3>(1, false, 1.0, 0.0, 0.0));

    return RunSim<3>(nodes, SimulatorType::CPU);
}


std::vector<std::array<float, 3>> TwoParticlesOutOfRange3DGPU() {

    std::vector<Node<3>*> nodes;
    nodes.push_back(new Node<3>(0, false, -1.0, -0.0, 0.0));
    nodes.push_back(new Node<3>(1, false, 1.0, 0.0, 0.0));

    return RunSim<3>(nodes, SimulatorType::GPU);
}


std::vector<std::array<float, 2>> TwoParticlesInRange2DCPU() {

    std::vector<Node<2>*> nodes;
    nodes.push_back(new Node<2>(0, false, -0.2, -0.2));
    nodes.push_back(new Node<2>(1, false, 0.2, 0.2));

    return RunSim<2>(nodes, SimulatorType::CPU);
}


std::vector<std::array<float, 2>> TwoParticlesInRange2DGPU() {

    std::vector<Node<2>*> nodes;
    nodes.push_back(new Node<2>(0, false, -0.2, -0.2));
    nodes.push_back(new Node<2>(1, false, 0.2, 0.2));

    return RunSim<2>(nodes, SimulatorType::GPU);
}

std::vector<std::array<float, 3>> TwoParticlesInRange3DCPU() {

    std::vector<Node<3>*> nodes;
    nodes.push_back(new Node<3>(0, false, -0.2, -0.2, -0.2));
    nodes.push_back(new Node<3>(1, false, 0.2, 0.2, 0.2));

    return RunSim<3>(nodes, SimulatorType::CPU);
}


std::vector<std::array<float, 3>> TwoParticlesInRange3DGPU() {

    std::vector<Node<3>*> nodes;
    nodes.push_back(new Node<3>(0, false, -0.2, -0.2, -0.2));
    nodes.push_back(new Node<3>(1, false, 0.2, 0.2, 0.2));

    return RunSim<3>(nodes, SimulatorType::GPU);
}

template<unsigned DIM>
double ComparePositions(const std::vector<std::array<float, DIM>>& v1, const std::vector<std::array<float, DIM>>& v2) {
    assert(v1.size() == v2.size());

    double totalDifference = 0.0;
    for (int i = 0; i < v1.size(); i++) {
        auto& p1 = v1[i];
        auto& p2 = v2[i];

        for (int dim = 0; dim < DIM; dim++) {
            totalDifference += std::abs(p2[dim] - p1[dim]);
        }
    }

    return totalDifference / v1.size();
}

double ValdiateTwoParticlesOutOfRange2D() {
    auto cpuResults = TwoParticlesOutOfRange2DCPU();
    auto gpuResults = TwoParticlesOutOfRange2DGPU();

    return ComparePositions<2>(cpuResults, gpuResults);
}

double ValdiateTwoParticlesInRange2D() {
    auto cpuResults = TwoParticlesInRange2DCPU();
    auto gpuResults = TwoParticlesInRange2DGPU();

    return ComparePositions<2>(cpuResults, gpuResults);
}

double ValdiateTwoParticlesOutOfRange3D() {
    auto cpuResults = TwoParticlesOutOfRange3DCPU();
    auto gpuResults = TwoParticlesOutOfRange3DGPU();

    return ComparePositions<3>(cpuResults, gpuResults);
}

double ValdiateTwoParticlesInRange3D() {
    auto cpuResults = TwoParticlesInRange3DCPU();
    auto gpuResults = TwoParticlesInRange3DGPU();

    return ComparePositions<3>(cpuResults, gpuResults);
}

double ValidateMultipleParticles(double box_size) {

    unsigned cells_across = box_size * 1.52;
    double scaling = box_size/(double(cells_across-1));

    // Create a simple 3D NodeBasedCellPopulation consisting of cells evenly spaced in a regular grid
    std::vector<Node<2>*> nodes;
    unsigned index = 0;
    for (unsigned i=0; i<cells_across; i++)
    {
        for (unsigned j=0; j<cells_across; j++)
        {
            nodes.push_back(new Node<2>(index, false,  (double) i * scaling , (double) j * scaling));
            index++;
        }
    }

    auto cpuResults = RunSim<2>(nodes, SimulatorType::CPU);

    nodes.clear();
    index = 0;
    for (unsigned i=0; i<cells_across; i++)
    {
        for (unsigned j=0; j<cells_across; j++)
        {
            nodes.push_back(new Node<2>(index, false,  (double) i * scaling , (double) j * scaling));
            index++;
        }
    }

    auto gpuResults = RunSim<2>(nodes, SimulatorType::GPU);

    return ComparePositions<2>(cpuResults, gpuResults);

}

void PerformForceValidation() {
    std::cout << "Starting force validation...\n";
    //auto multipleDiff = ValidateMultipleParticles(2);
    //std::cout << "Multiple particles diff: " << multipleDiff << "\n";
    auto outOfRangeDiff = ValdiateTwoParticlesOutOfRange2D();
    std::cout << "Out of range diff: " << outOfRangeDiff << "\n";
    auto inRangeDiff = ValdiateTwoParticlesInRange2D();
    std::cout << "In range diff: " << inRangeDiff << "\n";
    auto outOfRangeDiff3D = ValdiateTwoParticlesOutOfRange3D();
    std::cout << "Out of range 3D diff: " << outOfRangeDiff3D << "\n";
    auto inRangeDiff3D = ValdiateTwoParticlesInRange3D();
    std::cout << "In range 3D diff: " << inRangeDiff3D << "\n";
}

template<unsigned DIM>
void PerformBenchmarkSim(const double size_of_box, ResultsSet& results, SimulatorType type) {

    std::string typeString = type == SimulatorType::CPU ? "CPU" : "GPU";
    std::cout << "Starting " << typeString << " sim with box size: " << size_of_box << "\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    SimulationTime::Destroy();
    SimulationTime::Instance()->SetStartTime(0.0);

    unsigned cells_across = size_of_box * 1.52;
    double scaling = size_of_box/(double(cells_across-1));

    // Create a simple 3D NodeBasedCellPopulation consisting of cells evenly spaced in a regular grid
    std::vector<Node<DIM>*> nodes;
    unsigned index = 0;
    for (unsigned i=0; i<cells_across; i++)
    {
        for (unsigned j=0; j<cells_across; j++)
        {
            if constexpr (DIM == 2) {
                nodes.push_back(new Node<2>(index, false,  (double) i * scaling , (double) j * scaling));
                index++;
            }

            if constexpr (DIM == 3) {
                for (unsigned k = 0; k < cells_across; k++) {
                    nodes.push_back(new Node<3>(index, false,  (double) i * scaling , (double) j * scaling, (double) k * scaling));
                    index++;
                }
            }
        }
    }

    NodesOnlyMesh<DIM> mesh;
    mesh.ConstructNodesWithoutMesh(nodes, 1.5);

    std::vector<CellPtr> cells;
    MAKE_PTR(TransitCellProliferativeType, p_transit_type);
    CellsGenerator<UniformCellCycleModel, DIM> cells_generator;
    cells_generator.GenerateBasicRandom(cells, mesh.GetNumNodes(), p_transit_type);

    NodeBasedCellPopulation<DIM> node_based_cell_population(mesh, cells);

    // Set up cell-based simulation
    OffLatticeSimulation<DIM> simulator(node_based_cell_population);
    simulator.SetOutputDirectory("GPUNodeBased");
    simulator.SetSamplingTimestepMultiple(500);
    simulator.SetEndTime(END_TIME);

    MAKE_PTR(GeneralisedLinearSpringForce<DIM>, springForce);
    MAKE_PTR(GPUModifier<DIM>, gpuModifier);

    if (type == SimulatorType::CPU) {
        springForce->SetCutOffLength(1.5);
        simulator.AddForce(springForce);
    } else {
        simulator.AddSimulationModifier(gpuModifier);
    }

    // Run simulation
    simulator.Solve();

    // Avoid memory leak
    for (unsigned i=0; i<nodes.size(); i++)
    {
        delete nodes[i];
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double mechanics_time = 0.0;
    if (type == SimulatorType::GPU) {
        auto& timingResults = gpuModifier->mTimingInfo;

        for (auto& r : timingResults) {
            mechanics_time += r[5];
        }
    } else {
        for (auto& v : simulator.mechTimes) {
            mechanics_time += v;
        }
    }

    ResultsRow row;
    row.type = type == SimulatorType::CPU ? "cpu" : "gpu";
    row.dimensions = DIM == 2 ? "2D" : "3D";
    row.box_size = size_of_box;
    row.mechanics_time = mechanics_time;
    row.run_time = duration.count();
    results.push_back(row);

}

int main(int argc, char *argv[])
{
    // This sets up PETSc and prints out copyright information, etc.
    ExecutableSupport::StandardStartup(&argc, &argv);
    // Perf benchmark
    std::vector<double> box_sizes = {10.0, 20.0, 30.0, 40.0, 50.0};//, 100.0, 200.0, 300.0};//, 500.0, 750.0, 1000.0};
    std::vector<double> box_sizes_3D = {3.0, 5.0, 10.0};//, 20.0, 30.0};//, 100.0, 200.0, 300.0};//, 500.0, 750.0, 1000.0};
    ResultsSet results;
    //for (auto box_size : box_sizes) {
    //    PerformBenchmarkSim<2>(box_size, results, SimulatorType::CPU);
    //    PerformBenchmarkSim<2>(box_size, results, SimulatorType::GPU);
    //}
    for (auto box_size : box_sizes_3D) {
        PerformBenchmarkSim<3>(box_size, results, SimulatorType::CPU);
        PerformBenchmarkSim<3>(box_size, results, SimulatorType::GPU);
    }
    WriteResultsToFile(results, "results.txt");

    // Validation
    //PerformForceValidation();
}
